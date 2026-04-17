/**
 * @file    order_book.h
 * @brief   Architectural skeleton for a high-performance Limit Order Book (LOB)
 *          matching engine.
 *
 * =============================================================================
 * SYSTEM OVERVIEW & DATA FLOW
 * =============================================================================
 *
 *  A Limit Order Book is the core data structure of any electronic exchange.
 *  It maintains two sorted collections of resting orders:
 *    - BID side: Buyers, sorted DESCENDING by price (best bid = highest price).
 *    - ASK side: Sellers, sorted ASCENDING by price (best ask = lowest price).
 *
 *  A "match" occurs when the best bid price >= best ask price (price crossing).
 *
 *  DATA FLOW:
 *
 *    [Incoming Order]
 *          |
 *          v
 *    addOrder(order)
 *          |
 *          |--- Assigns unique ID & timestamp
 *          |--- Inserts into PriceLevel doubly linked list   <- O(1)
 *          |--- Registers node ptr in order_map (hash map)  <- O(1) avg
 *          |--- Inserts PriceLevel into bid_map / ask_map   <- O(log N) levels
 *          |
 *          v
 *    matchOrders()  [called after every addOrder]
 *          |
 *          |--- Peeks at best_bid (rbegin) and best_ask (begin)
 *          |--- If bid.price >= ask.price  --> MATCH
 *          |--- Executes trade, reduces quantities
 *          |--- Removes filled orders from list & hash map  <- O(1)
 *          |--- Removes empty PriceLevels from sorted map   <- O(log N)
 *          |
 *          v
 *    cancelOrder(order_id)
 *          |
 *          |--- Lookup node ptr in order_map                <- O(1) avg
 *          |--- Splice node out of doubly linked list       <- O(1)
 *          |--- Erase from order_map                        <- O(1) avg
 *          |--- If PriceLevel empty, erase from sorted map  <- O(log N)
 *
 * =============================================================================
 * KEY COMPLEXITY TARGETS (per operation)
 * =============================================================================
 *
 *  addOrder()     : O(log P)  where P = number of distinct price levels
 *  cancelOrder()  : O(log P)  — O(1) list splice + O(log P) level cleanup
 *  matchOrders()  : O(K log P) where K = number of fills in one matching pass
 *
 *  The dominant cost is always the sorted map (price level tree), not the
 *  per-order structures. In practice, P << N (total orders), keeping the
 *  tree shallow.
 *
 * =============================================================================
 */

#pragma once

#include <cstdint>
#include <chrono>
#include <string>
#include <map>
#include <unordered_map>
#include <list>
#include <optional>
#include <functional>
#include <stdexcept>
#include <memory>

namespace lob {

// =============================================================================
// SECTION 1 — FUNDAMENTAL TYPES & ENUMERATIONS
// =============================================================================

/// Unique identifier for every order in the system.
using OrderId    = uint64_t;

/// Price represented as a fixed-point integer (e.g., cents) to avoid
/// floating-point comparison hazards. Multiply raw price by 100 before use.
using Price      = int64_t;

/// Quantity in integer units (e.g., number of shares or contracts).
using Quantity   = uint64_t;

/// High-resolution timestamp sourced from a monotonic clock.
using Timestamp  = std::chrono::time_point<std::chrono::steady_clock>;


/**
 * @brief Which side of the book this order rests on.
 *
 *  BID  — a buyer willing to pay up to `price`
 *  ASK  — a seller willing to accept at least `price`
 */
enum class Side : uint8_t {
    BID,
    ASK
};


/**
 * @brief Execution type of an incoming order.
 *
 *  LIMIT  — rests in the book at a specific price if not immediately matched.
 *  MARKET — matches at any available price; never rests in the book.
 *           (Market orders are modelled as LIMIT orders with a sentinel price:
 *            INT64_MAX for buys, 0 for sells, so they always cross the spread.)
 */
enum class OrderType : uint8_t {
    LIMIT,
    MARKET
};


/**
 * @brief Lifecycle state of an order.
 *
 *  OPEN         — active and resting in the book.
 *  PARTIALLY_FILLED — has been matched against one or more orders but has
 *                     remaining quantity.
 *  FILLED       — fully matched; removed from the book.
 *  CANCELLED    — explicitly cancelled by the submitter.
 */
enum class OrderStatus : uint8_t {
    OPEN,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED
};


// =============================================================================
// SECTION 2 — ORDER STRUCT
// =============================================================================

/**
 * @brief Represents a single resting or incoming limit order.
 *
 *  Stored DIRECTLY inside a std::list<Order> node embedded in each PriceLevel.
 *  This means the order's memory lives within the price level's list, and we
 *  hold iterators (pointers to nodes) for O(1) splice/erasure.
 *
 *  Members are laid out to avoid padding waste on a 64-bit system:
 *    8B id | 8B price | 8B quantity | 8B filled_qty | 8B timestamp |
 *    1B side | 1B type | 1B status | (5B pad)
 */
struct Order {
    OrderId    id;                ///< Globally unique order identifier.
    Price      price;             ///< Limit price (fixed-point, e.g., cents).
    Quantity   quantity;          ///< Original submitted quantity.
    Quantity   filled_quantity;   ///< Quantity matched so far; 0 on creation.
    Timestamp  timestamp;         ///< Wall-clock time of submission (for FIFO priority).
    Side       side;              ///< BID or ASK.
    OrderType  type;              ///< LIMIT or MARKET.
    OrderStatus status;           ///< Current lifecycle state.

    /// Remaining unfilled quantity — the "live" quantity in the book.
    /// Inline for hot-path calculations; avoids repeated subtraction.
    [[nodiscard]] inline Quantity remaining() const noexcept {
        return quantity - filled_quantity;
    }

    /// An order is "dead" once it cannot participate in further matching.
    [[nodiscard]] inline bool is_active() const noexcept {
        return status == OrderStatus::OPEN ||
               status == OrderStatus::PARTIALLY_FILLED;
    }
};


// =============================================================================
// SECTION 3 — PRICE LEVEL
// =============================================================================

/**
 * @brief All resting orders at a single price point on one side of the book.
 *
 *  DESIGN RATIONALE — std::list<Order> (doubly linked list):
 *  ─────────────────────────────────────────────────────────
 *  ┌──────────────────────────────────────────────────────────┐
 *  │  Price Level $150.00 (BID side)                          │
 *  │                                                          │
 *  │  [Order A, qty 100] <-> [Order B, qty 250] <-> [Order C] │
 *  │       ↑ FIFO front (oldest, matched first)               │
 *  └──────────────────────────────────────────────────────────┘
 *
 *  - Insertion at back  : O(1)  — new orders join the queue at the tail.
 *  - Match from front   : O(1)  — oldest order at head is consumed first.
 *  - Cancellation       : O(1)  — given an iterator, list::erase is O(1)
 *                                  because no shifting, just pointer relinking.
 *
 *  A plain std::vector would give O(N) cancellation (element shift).
 *  A std::deque would give O(1) front pop but still O(N) mid-cancel.
 *  The linked list + external iterator map is the canonical solution.
 */
struct PriceLevel {
    Price              price;        ///< The price this level represents.
    Quantity           total_qty;    ///< Sum of all remaining() quantities here.
                                     ///  Maintained incrementally — O(1) updates.
    std::list<Order>   orders;       ///< FIFO queue of resting orders.

    // Convenience: aggregate visible volume at this level.
    // Used by market data feeds (Level 2 quotes).
    [[nodiscard]] bool empty() const noexcept { return orders.empty(); }
};

/// An iterator into PriceLevel::orders — this is the "pointer" we store in the
/// lookup map to enable O(1) cancellation without scanning the list.
using OrderIterator = std::list<Order>::iterator;


// =============================================================================
// SECTION 4 — TRADE / EXECUTION REPORT
// =============================================================================

/**
 * @brief Immutable record of a matched trade between two orders.
 *
 *  Generated by matchOrders() for each fill. In a production system this
 *  would be published to a trade feed, risk engine, and settlement layer.
 */
struct Trade {
    OrderId   bid_order_id;   ///< The resting or aggressing buy order.
    OrderId   ask_order_id;   ///< The resting or aggressing sell order.
    Price     exec_price;     ///< Price at which the fill occurred.
                              ///  Convention: resting order's price wins.
    Quantity  exec_quantity;  ///< Number of units matched in this fill.
    Timestamp exec_time;      ///< Timestamp of execution.
};

/// Callback invoked each time a trade is generated. Decouples the book from
/// downstream consumers (risk, PnL, market data). Avoids virtual dispatch.
using TradeCallback = std::function<void(const Trade&)>;


// =============================================================================
// SECTION 5 — ORDER BOOK CLASS
// =============================================================================

/**
 * @class OrderBook
 * @brief Central matching engine for a single trading instrument.
 *
 * INTERNAL DATA STRUCTURES:
 * ─────────────────────────
 *
 * (A) bid_map / ask_map  — std::map<Price, PriceLevel>
 *     ┌─────────────────────────────────────────────────────────┐
 *     │  std::map is a red-black tree (sorted by Price key).    │
 *     │                                                         │
 *     │  bid_map: sorted ASCENDING, best bid = rbegin()         │
 *     │  ask_map: sorted ASCENDING, best ask = begin()          │
 *     │                                                         │
 *     │  Access to best price: O(1) via cached iterator.        │
 *     │  Insert / erase a price level: O(log P).                │
 *     │  P = number of distinct active price levels (typically  │
 *     │  small, << total order count N).                        │
 *     └─────────────────────────────────────────────────────────┘
 *
 * (B) order_map  — std::unordered_map<OrderId, OrderLocation>
 *     ┌─────────────────────────────────────────────────────────┐
 *     │  Hash map for O(1) average-case order lookup.           │
 *     │                                                         │
 *     │  Maps:  OrderId  →  { side, Price, OrderIterator }      │
 *     │                                                         │
 *     │  The OrderIterator directly addresses the node inside   │
 *     │  the PriceLevel's std::list — no secondary search.      │
 *     │                                                         │
 *     │  Lookup: O(1) avg  |  Insert: O(1) avg  |  Erase O(1)  │
 *     │  Worst case O(N) due to hash collisions — mitigated by  │
 *     │  a good hash (FNV-1a or absl::Hash in production).      │
 *     └─────────────────────────────────────────────────────────┘
 *
 *  COMBINED CANCELLATION PATH (the power of pairing these two):
 *
 *    cancelOrder(id):
 *      1. order_map[id]  → retrieves (side, price, list_iterator)   O(1)
 *      2. list.erase(iterator)                                       O(1)
 *      3. bid/ask_map[price].total_qty -= order.remaining()          O(log P)
 *      4. if PriceLevel empty → bid/ask_map.erase(price)            O(log P)
 *
 *    Net: O(log P) — the tree lookup for level cleanup is the bottleneck.
 */
class OrderBook {
public:

    // -------------------------------------------------------------------------
    // Construction & Configuration
    // -------------------------------------------------------------------------

    /**
     * @param instrument_id   Symbol / ISIN this book manages.
     * @param on_trade        Callback fired on every execution.
     *                        Pass a no-op lambda if unused.
     */
    explicit OrderBook(std::string instrument_id,
                       TradeCallback on_trade = nullptr);

    // Non-copyable — an order book is a unique, stateful resource.
    OrderBook(const OrderBook&)            = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    // Movable for container storage.
    OrderBook(OrderBook&&)            noexcept = default;
    OrderBook& operator=(OrderBook&&) noexcept = default;

    ~OrderBook() = default;


    // -------------------------------------------------------------------------
    // Core Mutating Operations
    // -------------------------------------------------------------------------

    /**
     * @brief Submit a new order to the book.
     *
     *  STEPS:
     *    1. Stamp the order (ID, timestamp).
     *    2. For MARKET orders: set sentinel price, skip to matching.
     *    3. For LIMIT orders: attempt immediate match (aggressive order).
     *       If unfilled quantity remains → insert as resting order.
     *    4. Call matchOrders() to resolve any crossing.
     *
     *  COMPLEXITY: O(K log P)  where K = fills generated, P = price levels.
     *              Resting-only (no match): O(log P) for tree insert.
     *
     * @param order   Fully constructed Order; id and timestamp will be
     *                overwritten with server-assigned values.
     * @returns       The server-assigned OrderId for tracking.
     * @throws        std::invalid_argument if quantity == 0.
     */
    OrderId addOrder(Order order);


    /**
     * @brief Cancel a resting order by its ID.
     *
     *  Uses order_map for O(1) node lookup, then O(1) list splice,
     *  then O(log P) tree cleanup if the price level becomes empty.
     *
     *  COMPLEXITY: O(log P) amortised.
     *
     * @param order_id   ID previously returned by addOrder().
     * @returns          true if cancelled; false if not found / already dead.
     */
    bool cancelOrder(OrderId order_id);


    /**
     * @brief Modify the quantity of a resting order (partial cancel).
     *
     *  Reducing quantity is safe and preserves time priority.
     *  Increasing quantity = cancel + resubmit (loses queue position).
     *  This method enforces that invariant internally.
     *
     *  COMPLEXITY: O(log P)  for reduce;  O(K log P) for increase (re-insert).
     *
     * @param order_id    Target order.
     * @param new_qty     Desired new total quantity (must be > filled_qty).
     * @returns           true on success.
     * @throws            std::invalid_argument if new_qty <= filled_qty.
     */
    bool modifyOrder(OrderId order_id, Quantity new_qty);


    // -------------------------------------------------------------------------
    // Matching Engine — called internally, exposed for testing
    // -------------------------------------------------------------------------

    /**
     * @brief Attempt to match the current best bid against the best ask.
     *
     *  ALGORITHM (continuous double auction):
     *    while (best_bid.price >= best_ask.price):
     *        exec_qty  = min(bid.remaining(), ask.remaining())
     *        exec_price = resting order's price (price-time priority)
     *        emit Trade(exec_qty, exec_price)
     *        reduce quantities; remove filled orders; clean empty levels
     *
     *  COMPLEXITY: O(K log P)  where K = number of individual fills.
     *              Each fill: O(1) list pop + O(log P) potential level erase.
     *
     *  NOTE: For bulk ingestion (e.g., replay), call addOrder() in a loop
     *  and matching is triggered automatically — no manual invocation needed.
     */
    void matchOrders();


    // -------------------------------------------------------------------------
    // Read-Only Accessors (market data / introspection)
    // -------------------------------------------------------------------------

    /**
     * @brief Returns the current best bid price, or std::nullopt if book empty.
     *  O(1) — rbegin() on std::map is a cached pointer to the rightmost node.
     */
    [[nodiscard]] std::optional<Price> bestBid() const noexcept;

    /**
     * @brief Returns the current best ask price, or std::nullopt if book empty.
     *  O(1) — begin() on std::map.
     */
    [[nodiscard]] std::optional<Price> bestAsk() const noexcept;

    /**
     * @brief Mid-price = (best_bid + best_ask) / 2.0.
     *  Returns nullopt if either side is empty (no tradeable spread).
     *  O(1).
     */
    [[nodiscard]] std::optional<double> midPrice() const noexcept;

    /**
     * @brief Bid-ask spread in fixed-point units.
     *  O(1).
     */
    [[nodiscard]] std::optional<Price> spread() const noexcept;

    /**
     * @brief Total visible quantity resting at a given price on a given side.
     *  O(log P) — single tree lookup.
     */
    [[nodiscard]] Quantity volumeAtPrice(Side side, Price price) const noexcept;

    /**
     * @brief Look up a live order by ID.
     *  O(1) avg — hash map lookup.
     *  Returns nullptr if order is not active (filled, cancelled, not found).
     */
    [[nodiscard]] const Order* findOrder(OrderId order_id) const noexcept;

    /**
     * @brief Total number of resting orders across both sides.
     *  O(1) — maintained as an atomic counter.
     */
    [[nodiscard]] std::size_t totalOrderCount() const noexcept;

    /// Read-only access to the full bid price ladder (for Level 2 data feeds).
    [[nodiscard]] const std::map<Price, PriceLevel>& bidLadder() const noexcept;

    /// Read-only access to the full ask price ladder.
    [[nodiscard]] const std::map<Price, PriceLevel>& askLadder() const noexcept;


    // -------------------------------------------------------------------------
    // Diagnostics / Testing Hooks
    // -------------------------------------------------------------------------

    /**
     * @brief Validates internal consistency of all data structures.
     *
     *  Checks:
     *    - order_map size == sum of all list sizes across both sides.
     *    - PriceLevel::total_qty == sum of remaining() for all orders in it.
     *    - No filled/cancelled orders remain in any list.
     *
     *  COMPLEXITY: O(N)  — full scan; call only in debug/test builds.
     *  @returns  true if invariants hold; false (with stderr log) otherwise.
     */
    [[nodiscard]] bool assertInvariants() const;


private:

    // -------------------------------------------------------------------------
    // PRIVATE TYPES
    // -------------------------------------------------------------------------

    /**
     * @brief Index entry stored per order in the lookup hash map.
     *
     *  Bundles everything needed to locate and splice out an order in O(1):
     *    - `side`     → which map (bid_map or ask_map) to touch
     *    - `price`    → which PriceLevel within that map
     *    - `it`       → direct iterator into that PriceLevel's list
     */
    struct OrderLocation {
        Side          side;
        Price         price;
        OrderIterator it;     ///< Direct pointer into PriceLevel::orders list.
    };


    // -------------------------------------------------------------------------
    // PRIVATE DATA MEMBERS
    // -------------------------------------------------------------------------

    std::string   instrument_id_;   ///< Symbol this book is responsible for.
    TradeCallback on_trade_;        ///< Downstream trade event handler.
    OrderId       next_order_id_;   ///< Monotonically increasing ID counter.
    std::size_t   active_orders_;   ///< Cached count for O(1) totalOrderCount().


    /**
     * BID side price ladder.
     *
     *  std::map<Price, PriceLevel>  — red-black tree, sorted ASCENDING.
     *
     *  Best bid = highest price = rbegin().
     *  We use rbegin() rather than a reverse_map to avoid maintaining two
     *  separate trees; rbegin() is O(1) on std::map.
     *
     *  ALTERNATIVE CONSIDERED: std::priority_queue — gives O(1) best-bid
     *  peek but O(N) cancellation and no iteration. Rejected.
     */
    std::map<Price, PriceLevel> bid_map_;


    /**
     * ASK side price ladder.
     *
     *  std::map<Price, PriceLevel>  — red-black tree, sorted ASCENDING.
     *
     *  Best ask = lowest price = begin().
     */
    std::map<Price, PriceLevel> ask_map_;


    /**
     * Global order lookup index.
     *
     *  std::unordered_map<OrderId, OrderLocation>
     *
     *  This is the "secret weapon" for O(1) cancellation.
     *  Without it, cancelling an order requires:
     *    O(log P) tree lookup  +  O(N/P) linear scan through the price level.
     *  With it, the scan is eliminated — we jump straight to the list node.
     *
     *  Reserve capacity in the constructor to minimise rehashing in hot paths.
     *  Expected load factor < 0.75 (std::unordered_map default threshold).
     */
    std::unordered_map<OrderId, OrderLocation> order_map_;


    // -------------------------------------------------------------------------
    // PRIVATE HELPER METHODS
    // -------------------------------------------------------------------------

    /**
     * @brief Route an order to the correct side's price map and insert it.
     *
     *  Creates a new PriceLevel node if none exists at this price.
     *  Registers the resulting list iterator in order_map_.
     *
     *  COMPLEXITY: O(log P) for map::operator[].
     */
    void insertResting(Order&& order);


    /**
     * @brief Remove a fully filled or cancelled order from all structures.
     *
     *  Steps:
     *    1. Erase list node via stored iterator   O(1)
     *    2. Decrement PriceLevel::total_qty        O(1)
     *    3. If level empty → erase from bid/ask_map O(log P)
     *    4. Erase from order_map_                  O(1)
     *
     *  COMPLEXITY: O(log P) amortised.
     */
    void removeOrder(const OrderLocation& loc);


    /**
     * @brief Emit a trade and update both participating orders' quantities.
     *
     *  Updates filled_quantity and status on both Order objects.
     *  Fires the on_trade_ callback (if registered) synchronously.
     *  Caller is responsible for removing fully-filled orders afterward.
     *
     *  COMPLEXITY: O(1).
     */
    void executeFill(Order& bid_order, Order& ask_order,
                     Price exec_price, Quantity exec_qty);


    /**
     * @brief Compute the execution price under price-time priority rules.
     *
     *  Convention: the *resting* (passive) order's price is used.
     *  If both are aggressive (rare in single-threaded engines), the midpoint
     *  or the earlier order's price can be used — configurable via policy.
     *
     *  COMPLEXITY: O(1).
     */
    [[nodiscard]] static Price resolveExecPrice(const Order& resting,
                                                const Order& aggressor) noexcept;
};


// =============================================================================
// SECTION 6 — MULTI-INSTRUMENT ENGINE (OPTIONAL EXTENSION POINT)
// =============================================================================

/**
 * @class MatchingEngine
 * @brief Manages a collection of OrderBook instances, one per instrument.
 *
 *  Thin orchestration layer — routes incoming orders to the correct book,
 *  manages lifecycle, and aggregates trade reporting.
 *
 *  COMPLEXITY:
 *    routeOrder()   : O(1) avg (hash map instrument lookup) + O(K log P) match.
 *    instrument count has NO effect on per-book performance.
 */
class MatchingEngine {
public:
    explicit MatchingEngine(TradeCallback global_trade_cb = nullptr);

    /**
     * @brief Register a new tradeable instrument.
     *  Creates an empty OrderBook for this symbol.
     *  O(1) avg.
     */
    void addInstrument(const std::string& instrument_id);

    /**
     * @brief Route and process an incoming order for a given instrument.
     *  O(1) instrument lookup + O(K log P) matching.
     *  @throws std::out_of_range if instrument not registered.
     */
    OrderId submitOrder(const std::string& instrument_id, Order order);

    /**
     * @brief Cancel an order on a specific instrument's book.
     *  O(1) instrument lookup + O(log P) cancel.
     */
    bool cancelOrder(const std::string& instrument_id, OrderId order_id);

    /**
     * @brief Direct access to a specific book (e.g., for market data queries).
     *  O(1) avg.
     */
    [[nodiscard]] const OrderBook* getBook(const std::string& instrument_id) const noexcept;

private:
    /// Hash map from instrument symbol → OrderBook.
    /// Using unique_ptr to keep books stable under rehashing.
    std::unordered_map<std::string, std::unique_ptr<OrderBook>> books_;
    TradeCallback global_trade_cb_;
};

} // namespace lob