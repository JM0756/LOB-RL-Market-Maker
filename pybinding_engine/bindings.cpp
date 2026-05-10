/**
 * @file    bindings.cpp
 * @brief   Pybind11 wrapper exposing the lob::OrderBook C++ matching engine
 *          to Python. Intended for use in RL market-making environments where
 *          the agent interacts with the book via a Gymnasium-style step loop.
 *
 * =============================================================================
 * DESIGN DECISIONS
 * =============================================================================
 *
 *  1. NO TRADE CALLBACK IN PYTHON CONSTRUCTOR
 *     The TradeCallback is a hot-path synchronous hook. Crossing the
 *     Python/C++ boundary on every fill would destroy latency characteristics.
 *     Instead, we collect fills into a thread-safe internal buffer and expose
 *     a drain_trades() method that the Python env can call once per step.
 *     This keeps the matching loop 100% in C++ and batches the GIL crossing.
 *
 *  2. std::optional → Python None
 *     pybind11 automatically converts std::optional<T> to either a Python
 *     value of type T or None. No manual wrapping needed.
 *
 *  3. std::vector<Quantity> → Python list
 *     pybind11 automatically converts std::vector<T> to a Python list.
 *     If the agent uses NumPy, it can call np.array(book.get_top_volumes(...))
 *     cheaply since no copy of the underlying data is needed beyond the list.
 *
 *  4. ENUMS as py::enum_
 *     We use py::enum_<> with .export_values() so Python code can write both:
 *       lob.Side.BID          ← scoped access (preferred)
 *       lob.BID               ← module-level access (for brevity in notebooks)
 *
 *  5. ORDER STRUCT — keyword constructor
 *     We expose Order via py::init<>() with keyword arguments matching the
 *     C++ struct fields. The RL agent typically constructs orders like:
 *       order = lob.Order(price=15000, quantity=100, side=lob.Side.BID,
 *                         type=lob.OrderType.LIMIT)
 *     Fields that are server-stamped (id, timestamp, filled_quantity, status)
 *     have sensible defaults and will be overwritten by addOrder() anyway.
 *
 *  6. GIL MANAGEMENT
 *     The matching engine is single-threaded by design (one Python thread
 *     drives the env). We do NOT release the GIL inside C++ methods because:
 *       a) The callback buffer write is not thread-safe without a mutex.
 *       b) The RL training loop is already GIL-bound via PyTorch/JAX.
 *     If you add multi-threading later, wrap hot methods with
 *     py::call_guard<py::gil_scoped_release>().
 *
 * =============================================================================
 * PYTHON USAGE EXAMPLE
 * =============================================================================
 *
 *   import lob_engine as lob
 *   import numpy as np
 *
 *   book = lob.OrderBook("AAPL")
 *
 *   # Place resting liquidity
 *   bid_id = book.add_order(lob.Order(
 *       price=15000, quantity=100,
 *       side=lob.Side.BID, type=lob.OrderType.LIMIT))
 *
 *   ask_id = book.add_order(lob.Order(
 *       price=15010, quantity=100,
 *       side=lob.Side.ASK, type=lob.OrderType.LIMIT))
 *
 *   # Consume the trade buffer after each step
 *   trades = book.drain_trades()   # list[lob.Trade]
 *
 *   # Build the RL observation vector
 *   bid_vols = np.array(book.get_top_volumes(lob.Side.BID, 5), dtype=np.float32)
 *   ask_vols = np.array(book.get_top_volumes(lob.Side.ASK, 5), dtype=np.float32)
 *   obs = np.concatenate([bid_vols, ask_vols])
 *
 * =============================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>           // std::vector, std::optional  ↔  Python list / None
#include <pybind11/functional.h>    // std::function  ↔  Python callable (used internally)
#include <pybind11/chrono.h>        // std::chrono::time_point  ↔  Python datetime

#include "OrderBook.H"

#include <mutex>
#include <vector>
#include <memory>
#include <string>

namespace py = pybind11;

// =============================================================================
// INTERNAL HELPER — PyOrderBook
// =============================================================================
//
//  A thin adapter that owns an lob::OrderBook and captures all Trade events
//  into an internal buffer. Python calls drain_trades() to consume the buffer.
//
//  Why not expose lob::OrderBook directly?
//  ─────────────────────────────────────────
//  The C++ TradeCallback fires *synchronously inside the matching loop*.
//  If we handed a Python callable straight to the C++ constructor, every fill
//  would require acquiring the GIL, constructing Python objects, dispatching
//  the callable, and releasing the GIL — all while the matching loop is mid-
//  execution. That would be unsafe and catastrophically slow.
//
//  Instead, we install a plain C++ lambda that pushes Trade structs into a
//  std::vector<lob::Trade>. After the Python step() call returns, the agent
//  calls drain_trades() once to collect all fills from that step. This keeps
//  the critical matching path entirely in C++.

class PyOrderBook {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------
    //
    //  Python signature:  OrderBook(instrument_id: str) -> OrderBook
    //
    //  The TradeCallback is intentionally hidden from Python. We install our
    //  own lambda that captures `trade_buffer_` by reference.
    explicit PyOrderBook(const std::string& instrument_id)
        : book_(
            instrument_id,
            // C++ lambda — zero GIL crossings, no Python objects constructed.
            [this](const lob::Trade& t) {
                trade_buffer_.push_back(t);
            }
          )
    {}

    // -------------------------------------------------------------------------
    // Core Mutating Operations (thin forwards)
    // -------------------------------------------------------------------------

    lob::OrderId addOrder(lob::Order order) {
        return book_.addOrder(std::move(order));
    }

    bool cancelOrder(lob::OrderId order_id) {
        return book_.cancelOrder(order_id);
    }

    bool modifyOrder(lob::OrderId order_id, lob::Quantity new_qty) {
        return book_.modifyOrder(order_id, new_qty);
    }

    // -------------------------------------------------------------------------
    // Trade Buffer Drain
    // -------------------------------------------------------------------------
    //
    //  Returns all fills generated since the last call and clears the buffer.
    //  Typical RL usage:
    //
    //    trades = book.drain_trades()
    //    reward = sum(t.exec_quantity * t.exec_price for t in trades)

    std::vector<lob::Trade> drainTrades() {
        std::vector<lob::Trade> out;
        out.swap(trade_buffer_);    // O(1) swap — no element copying.
        return out;                 // NRVO ensures no additional copy.
    }

    // -------------------------------------------------------------------------
    // Read-Only Market Data Accessors (thin forwards)
    // -------------------------------------------------------------------------

    std::optional<lob::Price>  bestBid()  const noexcept { return book_.bestBid();  }
    std::optional<lob::Price>  bestAsk()  const noexcept { return book_.bestAsk();  }
    std::optional<double>      midPrice() const noexcept { return book_.midPrice(); }
    std::optional<lob::Price>  spread()   const noexcept { return book_.spread();   }

    lob::Quantity volumeAtPrice(lob::Side side, lob::Price price) const noexcept {
        return book_.volumeAtPrice(side, price);
    }

    std::size_t totalOrderCount() const noexcept {
        return book_.totalOrderCount();
    }

    // -------------------------------------------------------------------------
    // findOrder — O(1) order lookup by ID
    // -------------------------------------------------------------------------
    //
    //  Forwards directly to lob::OrderBook::findOrder(), which returns a raw
    //  pointer into the std::list<Order> node that lives inside the C++ book.
    //
    //  OWNERSHIP CONTRACT (critical for pybind11 binding):
    //  ────────────────────────────────────────────────────
    //  The Order object is owned exclusively by the C++ engine — specifically,
    //  it lives inside a std::list<Order> node within a PriceLevel, which is
    //  itself owned by PyOrderBook::book_. Python must NEVER free or take
    //  ownership of this memory. We enforce this by binding with:
    //
    //      py::return_value_policy::reference
    //
    //  This tells pybind11: "hand Python a view of the C++ object; do NOT
    //  generate a destructor call when the Python reference goes out of scope."
    //
    //  LIFETIME GUARANTEE:
    //  The returned pointer (and therefore the Python object it backs) is valid
    //  only as long as the order remains in the book (i.e., it has not been
    //  filled, cancelled, or had its list node erased). The caller must not
    //  cache the returned object across step() boundaries — re-query each time.
    //  Returning nullptr (→ Python None) is the engine's signal that the order
    //  is no longer live.
    //
    //  WHY NOT reference_internal?
    //  py::return_value_policy::reference_internal would additionally tie the
    //  lifetime of the returned Python object to the parent (the PyOrderBook),
    //  preventing the book from being garbage-collected while a dangling Order
    //  reference exists. That extra safety net is appropriate for long-lived
    //  cached references. Here the caller pattern is:
    //      order = book.find_order(oid)   # ephemeral — used and discarded
    //  So plain `reference` is the correct, lighter-weight choice.

    const lob::Order* findOrder(lob::OrderId order_id) const noexcept {
        return book_.findOrder(order_id);
    }

    // -------------------------------------------------------------------------
    // getTopVolumes — primary RL observation builder
    // -------------------------------------------------------------------------
    //
    //  Returns a fixed-length Python list of Quantity values (uint64).
    //  Convert to a NumPy array in Python for zero-copy tensor construction:
    //
    //    obs = np.array(book.get_top_volumes(lob.Side.BID, depth), dtype=np.float32)

    std::vector<lob::Quantity> getTopVolumes(lob::Side side, int depth) const {
        return book_.getTopVolumes(side, depth);
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    bool assertInvariants() const {
        return book_.assertInvariants();
    }

private:
    lob::OrderBook          book_;
    std::vector<lob::Trade> trade_buffer_;
};


// =============================================================================
// PYBIND11 MODULE DEFINITION
// =============================================================================
//
//  Module name: lob_engine
//  Import in Python:  import lob_engine as lob

PYBIND11_MODULE(lob_engine, m) {

    m.doc() =
        "lob_engine — High-performance C++ Limit Order Book matching engine.\n"
        "Designed for use as the market simulator in RL market-making agents.\n\n"
        "Quickstart:\n"
        "    import lob_engine as lob\n"
        "    book = lob.OrderBook('AAPL')\n"
        "    oid  = book.add_order(lob.Order(price=15000, quantity=100,\n"
        "                                    side=lob.Side.BID,\n"
        "                                    type=lob.OrderType.LIMIT))\n"
        "    obs  = book.get_top_volumes(lob.Side.BID, depth=5)\n";


    // =========================================================================
    // ENUMERATIONS
    // =========================================================================

    // ── Side ─────────────────────────────────────────────────────────────────
    py::enum_<lob::Side>(m, "Side",
        "Which side of the book an order rests on.\n\n"
        "  BID — buyer, willing to pay up to `price`.\n"
        "  ASK — seller, willing to accept at least `price`.")
        .value("BID", lob::Side::BID)
        .value("ASK", lob::Side::ASK)
        .export_values();   // Makes lob.BID and lob.ASK available at module level.

    // ── OrderType ────────────────────────────────────────────────────────────
    py::enum_<lob::OrderType>(m, "OrderType",
        "Execution type of an incoming order.\n\n"
        "  LIMIT  — rests in the book at a specific price if not filled.\n"
        "  MARKET — matches at any price; uses a sentinel price internally.")
        .value("LIMIT",  lob::OrderType::LIMIT)
        .value("MARKET", lob::OrderType::MARKET)
        .export_values();

    // ── OrderStatus ──────────────────────────────────────────────────────────
    py::enum_<lob::OrderStatus>(m, "OrderStatus",
        "Lifecycle state of an order.\n\n"
        "  OPEN             — active and resting in the book.\n"
        "  PARTIALLY_FILLED — partially matched, remaining qty still resting.\n"
        "  FILLED           — fully matched; removed from the book.\n"
        "  CANCELLED        — explicitly cancelled.")
        .value("OPEN",             lob::OrderStatus::OPEN)
        .value("PARTIALLY_FILLED", lob::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED",           lob::OrderStatus::FILLED)
        .value("CANCELLED",        lob::OrderStatus::CANCELLED)
        .export_values();


    // =========================================================================
    // STRUCT: Order
    // =========================================================================
    //
    //  Exposed as a plain Python class with read/write attributes.
    //  The keyword-argument constructor lets the agent write idiomatic Python:
    //
    //    order = lob.Order(
    //        price    = 15005,
    //        quantity = 200,
    //        side     = lob.Side.ASK,
    //        type     = lob.OrderType.LIMIT,
    //    )
    //
    //  Fields that the engine stamps on submission (id, timestamp,
    //  filled_quantity, status) are given sensible defaults so the caller
    //  doesn't need to supply them.

    py::class_<lob::Order>(m, "Order",
        "A single resting or incoming limit/market order.\n\n"
        "Server-stamped fields (id, timestamp, filled_quantity, status) are\n"
        "overwritten by OrderBook.add_order() and can be left at their defaults.\n\n"
        "Prices are fixed-point integers (e.g., cents). Multiply the raw dollar\n"
        "price by 100 before passing it to the engine.")
        // ── Keyword-argument constructor ──────────────────────────────────────
        .def(py::init([](
                lob::Price       price,
                lob::Quantity    quantity,
                lob::Side        side,
                lob::OrderType   type,
                lob::OrderId     id,
                lob::Quantity    filled_quantity,
                lob::OrderStatus status
            ) {
                lob::Order o;
                o.id              = id;
                o.price           = price;
                o.quantity        = quantity;
                o.filled_quantity = filled_quantity;
                o.timestamp       = std::chrono::steady_clock::time_point{};
                o.side            = side;
                o.type            = type;
                o.status          = status;
                return o;
            }),
            py::arg("price"),
            py::arg("quantity"),
            py::arg("side"),
            py::arg("type")            = lob::OrderType::LIMIT,
            py::arg("id")              = lob::OrderId{0},
            py::arg("filled_quantity") = lob::Quantity{0},
            py::arg("status")          = lob::OrderStatus::OPEN,
            "Construct an Order.\n\n"
            "Args:\n"
            "    price:           Fixed-point limit price (e.g., cents).\n"
            "    quantity:        Number of shares/contracts.\n"
            "    side:            Side.BID or Side.ASK.\n"
            "    type:            OrderType.LIMIT (default) or OrderType.MARKET.\n"
            "    id:              Leave as 0; overwritten by add_order().\n"
            "    filled_quantity: Leave as 0; maintained by the engine.\n"
            "    status:          Leave as OPEN; maintained by the engine.")
        // ── Attributes ────────────────────────────────────────────────────────
        .def_readwrite("id",              &lob::Order::id)
        .def_readwrite("price",           &lob::Order::price)
        .def_readwrite("quantity",        &lob::Order::quantity)
        .def_readwrite("filled_quantity", &lob::Order::filled_quantity)
        .def_readwrite("side",            &lob::Order::side)
        .def_readwrite("type",            &lob::Order::type)
        .def_readwrite("status",          &lob::Order::status)
        // ── Computed properties ───────────────────────────────────────────────
        .def("remaining", &lob::Order::remaining,
            "Remaining unfilled quantity (quantity - filled_quantity).")
        .def("is_active", &lob::Order::is_active,
            "True if the order is OPEN or PARTIALLY_FILLED.")
        // ── __repr__ ─────────────────────────────────────────────────────────
        .def("__repr__", [](const lob::Order& o) {
            return "<lob.Order id=" + std::to_string(o.id)
                 + " side=" + (o.side == lob::Side::BID ? "BID" : "ASK")
                 + " price=" + std::to_string(o.price)
                 + " qty=" + std::to_string(o.quantity)
                 + " filled=" + std::to_string(o.filled_quantity)
                 + ">";
        });


    // =========================================================================
    // STRUCT: Trade
    // =========================================================================
    //
    //  Read-only record returned by drain_trades(). The agent uses these to
    //  compute step rewards, track PnL, or log the tape.

    py::class_<lob::Trade>(m, "Trade",
        "Immutable record of a single fill between a bid order and an ask order.\n\n"
        "Obtain via OrderBook.drain_trades() after each step.")
        .def_readonly("bid_order_id",  &lob::Trade::bid_order_id,
            "OrderId of the buy side.")
        .def_readonly("ask_order_id",  &lob::Trade::ask_order_id,
            "OrderId of the sell side.")
        .def_readonly("exec_price",    &lob::Trade::exec_price,
            "Fill price in fixed-point units (e.g., cents).")
        .def_readonly("exec_quantity", &lob::Trade::exec_quantity,
            "Number of shares/contracts filled.")
        .def_readonly("exec_time",     &lob::Trade::exec_time,
            "Monotonic timestamp of the fill (std::chrono::steady_clock).")
        .def("__repr__", [](const lob::Trade& t) {
            return "<lob.Trade qty=" + std::to_string(t.exec_quantity)
                 + " @ " + std::to_string(t.exec_price)
                 + " bid_id=" + std::to_string(t.bid_order_id)
                 + " ask_id=" + std::to_string(t.ask_order_id)
                 + ">";
        });


    // =========================================================================
    // CLASS: OrderBook
    // =========================================================================

    py::class_<PyOrderBook>(m, "OrderBook",
        "Single-instrument Limit Order Book matching engine.\n\n"
        "Prices are fixed-point integers (multiply raw dollar price × 100).\n\n"
        "Trade events are buffered internally and retrieved via drain_trades(),\n"
        "which returns all fills since the last call and clears the buffer.\n\n"
        "Example (RL step loop)::\n\n"
        "    book = lob.OrderBook('AAPL')\n"
        "    oid  = book.add_order(lob.Order(15000, 100, lob.Side.BID))\n"
        "    trades = book.drain_trades()\n"
        "    obs    = book.get_top_volumes(lob.Side.BID, depth=5)\n")

        // ── Constructor ───────────────────────────────────────────────────────
        .def(py::init<const std::string&>(),
            py::arg("instrument_id"),
            "Create an empty OrderBook for the given instrument symbol.\n\n"
            "Args:\n"
            "    instrument_id: Ticker symbol, e.g. 'AAPL'.")

        // ── Core Mutating Operations ───────────────────────────────────────────
        .def("add_order",
            &PyOrderBook::addOrder,
            py::arg("order"),
            "Submit a new order. Returns the server-assigned integer OrderId.\n\n"
            "The engine overwrites order.id and order.timestamp. Market orders\n"
            "are assigned a sentinel price (INT64_MAX for BID, 0 for ASK) and\n"
            "will never rest in the book.\n\n"
            "Args:\n"
            "    order: An lob.Order object.\n"
            "Returns:\n"
            "    int: The assigned OrderId (use this for cancel / modify).\n"
            "Raises:\n"
            "    ValueError: If order.quantity is 0.")

        .def("cancel_order",
            &PyOrderBook::cancelOrder,
            py::arg("order_id"),
            "Cancel a resting order by its OrderId.\n\n"
            "Args:\n"
            "    order_id: Integer ID returned by add_order().\n"
            "Returns:\n"
            "    bool: True if cancelled; False if not found or already dead.")

        .def("modify_order",
            &PyOrderBook::modifyOrder,
            py::arg("order_id"),
            py::arg("new_qty"),
            "Resize a resting order's quantity.\n\n"
            "Reducing preserves queue position (in-place, O(1)).\n"
            "Increasing cancels and resubmits — the order loses queue position.\n\n"
            "Args:\n"
            "    order_id: Integer ID returned by add_order().\n"
            "    new_qty:  New total quantity (must exceed filled_quantity).\n"
            "Returns:\n"
            "    bool: True on success; False if order not found.\n"
            "Raises:\n"
            "    ValueError: If new_qty <= filled_quantity.")

        // ── Trade Buffer ──────────────────────────────────────────────────────
        .def("drain_trades",
            &PyOrderBook::drainTrades,
            "Return and clear all Trade records buffered since the last call.\n\n"
            "Call once per environment step to collect fills and compute reward.\n\n"
            "Returns:\n"
            "    list[lob.Trade]: All fills generated since the last drain.\n"
            "                     Empty list if no trades occurred.")

        // ── Market Data Accessors ─────────────────────────────────────────────
        .def("best_bid",
            &PyOrderBook::bestBid,
            "Best bid price (fixed-point int), or None if the bid side is empty.")

        .def("best_ask",
            &PyOrderBook::bestAsk,
            "Best ask price (fixed-point int), or None if the ask side is empty.")

        .def("mid_price",
            &PyOrderBook::midPrice,
            "Mid-price as a float: (best_bid + best_ask) / 2.0.\n"
            "Returns None if either side is empty.")

        .def("spread",
            &PyOrderBook::spread,
            "Bid-ask spread in fixed-point units (best_ask - best_bid).\n"
            "Returns None if either side is empty.")

        .def("volume_at_price",
            &PyOrderBook::volumeAtPrice,
            py::arg("side"),
            py::arg("price"),
            "Total resting volume at a specific price level.\n\n"
            "Args:\n"
            "    side:  Side.BID or Side.ASK.\n"
            "    price: Fixed-point price (e.g., 15000 for $150.00).\n"
            "Returns:\n"
            "    int: Total quantity; 0 if no orders at that price.")

        .def("total_order_count",
            &PyOrderBook::totalOrderCount,
            "Total number of resting orders across both sides of the book.")

        // ── find_order ────────────────────────────────────────────────────────
        //
        //  RETURN VALUE POLICY: py::return_value_policy::reference
        //  ──────────────────────────────────────────────────────────
        //  findOrder() returns a raw `const Order*` that points directly into a
        //  std::list<Order> node inside the C++ engine. There are three possible
        //  ownership policies pybind11 could apply:
        //
        //  Policy                  | What it does
        //  ──────────────────────  | ────────────────────────────────────────────
        //  copy (default)          | Copies the Order into a new Python-owned object.
        //                          | Safe but WRONG here: nullptr cannot be copied,
        //                          | and the copy would not reflect live qty changes.
        //  take_ownership          | Python's GC would call delete on the pointer.
        //                          | CATASTROPHIC — the pointer is not heap-allocated
        //                          | independently; it's a node inside std::list.
        //  reference               | Python gets a non-owning view. C++ retains full
        //  ✓ CORRECT               | ownership. nullptr maps to Python None. The Order
        //                          | object reflects live state (e.g., remaining qty
        //                          | decreasing as partial fills occur).
        //  reference_internal      | Same as reference, but additionally prevents the
        //                          | parent (PyOrderBook) from being GC'd while the
        //                          | returned Python object is alive. Not needed here
        //                          | because callers use find_order() ephemerally.
        //
        //  USAGE CONTRACT IN MarketMakerEnv.py:
        //
        //      order = book.find_order(self._bid_order_id)
        //      if order is None:
        //          # Order fully consumed — clear our tracked ID.
        //          self._bid_order_id = None
        //      else:
        //          # Order still resting — order.remaining() gives live qty.
        //          print(order.remaining())
        //
        //  Do NOT store the returned object across step() calls. Once the C++
        //  engine removes the order (fill or cancel), the pointer is dangling.
        .def("find_order",
            &PyOrderBook::findOrder,
            py::arg("order_id"),
            py::return_value_policy::reference,
            "Look up a live resting order by its OrderId.\n\n"
            "Returns a *non-owning* reference to the Order object that lives\n"
            "inside the C++ engine. Python must not hold this reference across\n"
            "step() calls — re-query each time you need current state.\n\n"
            "Returns None if the order is not found (already filled, cancelled,\n"
            "or the ID was never issued). This is the primary way to detect\n"
            "whether a resting quote has been fully consumed:\n\n"
            "    order = book.find_order(bid_id)\n"
            "    if order is None:\n"
            "        bid_id = None  # fully filled or cancelled\n"
            "    else:\n"
            "        remaining_qty = order.remaining()  # live partial fill state\n\n"
            "Args:\n"
            "    order_id: Integer OrderId returned by a prior add_order() call.\n"
            "Returns:\n"
            "    lob.Order | None: Live order view, or None if not in the book.\n\n"
            "Warning:\n"
            "    The returned object is a direct reference into C++ memory.\n"
            "    Do not cache it; do not call it after cancel_order() or after\n"
            "    the order has been fully filled.")

        // ── getTopVolumes — RL Feature Builder ───────────────────────────────
        .def("get_top_volumes",
            &PyOrderBook::getTopVolumes,
            py::arg("side"),
            py::arg("depth"),
            "Return a fixed-length list of total volumes at the top price levels.\n\n"
            "'Top' means best-priced:\n"
            "  BID: highest price first (index 0 = best bid level).\n"
            "  ASK: lowest  price first (index 0 = best ask level).\n\n"
            "The list is always exactly `depth` elements long. If the book has\n"
            "fewer than `depth` active levels, the tail is zero-padded — ensuring\n"
            "a constant-dimension observation for the RL policy network.\n\n"
            "Typical usage::\n\n"
            "    import numpy as np\n"
            "    bid_vols = np.array(book.get_top_volumes(lob.Side.BID, 5),\n"
            "                        dtype=np.float32)\n"
            "    ask_vols = np.array(book.get_top_volumes(lob.Side.ASK, 5),\n"
            "                        dtype=np.float32)\n"
            "    obs = np.concatenate([bid_vols, ask_vols])  # shape: (10,)\n\n"
            "Args:\n"
            "    side:  Side.BID or Side.ASK.\n"
            "    depth: Number of price levels to include (>= 0).\n"
            "Returns:\n"
            "    list[int]: Volumes of length exactly `depth`, zero-padded if\n"
            "               the book has fewer levels than requested.")

        // ── Diagnostics ───────────────────────────────────────────────────────
        .def("assert_invariants",
            &PyOrderBook::assertInvariants,
            "Run O(N) internal consistency checks (debug/test builds only).\n\n"
            "Returns:\n"
            "    bool: True if all invariants hold; False with stderr diagnostics\n"
            "          if any inconsistency is detected.");

}   // PYBIND11_MODULE