/**
 * @file    order_book.cpp
 * @brief   Implementation of the high-performance Limit Order Book (LOB)
 *          matching engine declared in order_book.h.
 *
 * =============================================================================
 * READING GUIDE
 * =============================================================================
 *
 *  Functions are ordered by logical dependency, not alphabetically:
 *
 *    1.  OrderBook constructor / destructor
 *    2.  insertResting()      ← primitive: add one order to the book structures
 *    3.  removeOrder()        ← primitive: remove one order from all structures
 *    4.  executeFill()        ← primitive: record one fill between two orders
 *    5.  resolveExecPrice()   ← primitive: price-time priority price selection
 *    6.  addOrder()           ← public: entry point for new orders
 *    7.  matchOrders()        ← public: the core matching loop
 *    8.  cancelOrder()        ← public: cancel a resting order
 *    9.  modifyOrder()        ← public: resize a resting order
 *   10.  Read-only accessors  ← bestBid, bestAsk, midPrice, spread, ...
 *   10b. getTopVolumes()      ← NEW: fixed-depth volume vector for RL agents
 *   11.  assertInvariants()   ← debug/test consistency checker
 *   12.  MatchingEngine       ← multi-instrument orchestration layer
 *
 * =============================================================================
 */

#include "OrderBook.H"

#include <algorithm>    // std::min
#include <iostream>     // std::cerr (invariant checker only)
#include <cassert>      // assert()
#include <limits>       // std::numeric_limits (market order sentinel prices)
#include <stdexcept>    // std::invalid_argument, std::out_of_range

namespace lob {

// =============================================================================
// SECTION 1 — CONSTRUCTION
// =============================================================================

OrderBook::OrderBook(std::string instrument_id, TradeCallback on_trade)
    : instrument_id_(std::move(instrument_id))
    , on_trade_      (std::move(on_trade))
    , next_order_id_ (1)          // Start IDs at 1; 0 is a sentinel "null" ID.
    , active_orders_ (0)
{
    // Pre-allocate the hash map to hold ~1024 orders before its first rehash.
    // Rehashing invalidates NO iterators (unordered_map guarantees iterator
    // stability on rehash for the VALUES, but our iterators point into the
    // std::list nodes — those are separately allocated and never move).
    // This reservation purely reduces amortised insert cost.
    order_map_.reserve(1024);
}


// =============================================================================
// SECTION 2 — PRIVATE PRIMITIVE: insertResting()
// =============================================================================

void OrderBook::insertResting(Order&& order) {
    auto& target_map = (order.side == Side::BID) ? bid_map_ : ask_map_;

    PriceLevel& level = target_map[order.price];

    if (level.orders.empty()) {
        level.price = order.price;
    }

    level.total_qty += order.remaining();

    level.orders.push_back(std::move(order));

    OrderIterator it = std::prev(level.orders.end());

    order_map_.emplace(it->id, OrderLocation{ it->side, it->price, it });

    ++active_orders_;
}


// =============================================================================
// SECTION 3 — PRIVATE PRIMITIVE: removeOrder()
// =============================================================================

void OrderBook::removeOrder(const OrderLocation& loc) {
    auto& target_map = (loc.side == Side::BID) ? bid_map_ : ask_map_;

    auto map_it = target_map.find(loc.price);

    assert(map_it != target_map.end() && "OrderLocation points to nonexistent price level");

    PriceLevel& level = map_it->second;

    level.total_qty -= loc.it->remaining();

    const OrderId erased_id = loc.it->id;
    level.orders.erase(loc.it);

    if (level.empty()) {
        target_map.erase(map_it);
    }

    order_map_.erase(erased_id);

    --active_orders_;
}


// =============================================================================
// SECTION 4 — PRIVATE PRIMITIVE: executeFill()
// =============================================================================

void OrderBook::executeFill(Order& bid_order, Order& ask_order,
                            Price exec_price, Quantity exec_qty) {
    assert(exec_qty > 0);
    assert(exec_qty <= bid_order.remaining());
    assert(exec_qty <= ask_order.remaining());

    bid_order.filled_quantity += exec_qty;
    bid_order.status = (bid_order.remaining() == 0)
                       ? OrderStatus::FILLED
                       : OrderStatus::PARTIALLY_FILLED;

    ask_order.filled_quantity += exec_qty;
    ask_order.status = (ask_order.remaining() == 0)
                       ? OrderStatus::FILLED
                       : OrderStatus::PARTIALLY_FILLED;

    if (on_trade_) {
        Trade trade {
            bid_order.id,
            ask_order.id,
            exec_price,
            exec_qty,
            std::chrono::steady_clock::now()
        };
        on_trade_(trade);
    }
}


// =============================================================================
// SECTION 5 — PRIVATE PRIMITIVE: resolveExecPrice()
// =============================================================================

/*static*/
Price OrderBook::resolveExecPrice(const Order& resting,
                                  const Order& /*aggressor*/) noexcept {
    return resting.price;
}


// =============================================================================
// SECTION 6 — PUBLIC: addOrder()
// =============================================================================

OrderId OrderBook::addOrder(Order order) {
    if (order.quantity == 0) {
        throw std::invalid_argument("Order quantity must be greater than zero.");
    }

    order.id               = next_order_id_++;
    order.timestamp        = std::chrono::steady_clock::now();
    order.filled_quantity  = 0;
    order.status           = OrderStatus::OPEN;

    const OrderId assigned_id = order.id;

    if (order.type == OrderType::MARKET) {
        order.price = (order.side == Side::BID)
                      ? std::numeric_limits<Price>::max()
                      : Price{0};
    }

    insertResting(std::move(order));

    matchOrders();

    return assigned_id;
}


// =============================================================================
// SECTION 7 — PUBLIC: matchOrders()
// =============================================================================

void OrderBook::matchOrders() {
    while (!bid_map_.empty() && !ask_map_.empty()) {

        auto best_bid_it = bid_map_.rbegin();
        PriceLevel& bid_level = best_bid_it->second;

        auto best_ask_it = ask_map_.begin();
        PriceLevel& ask_level = best_ask_it->second;

        if (bid_level.price < ask_level.price) {
            break;
        }

        Order& bid_order = bid_level.orders.front();
        Order& ask_order = ask_level.orders.front();

        const bool bid_is_older   = (bid_order.id < ask_order.id);
        const Order& resting_order  = bid_is_older ? bid_order : ask_order;
        const Order& aggressing_order = bid_is_older ? ask_order : bid_order;

        const Price exec_price = resolveExecPrice(resting_order, aggressing_order);

        const Quantity exec_qty = std::min(bid_order.remaining(),
                                           ask_order.remaining());

        executeFill(bid_order, ask_order, exec_price, exec_qty);

        bid_level.total_qty -= exec_qty;
        ask_level.total_qty -= exec_qty;

        const bool bid_filled = (bid_order.status == OrderStatus::FILLED);
        const bool ask_filled = (ask_order.status == OrderStatus::FILLED);

        const OrderId bid_id = bid_order.id;
        const OrderId ask_id = ask_order.id;

        if (bid_filled) {
            const OrderLocation& bid_loc = order_map_.at(bid_id);
            removeOrder(bid_loc);
        }

        if (ask_filled) {
            const OrderLocation& ask_loc = order_map_.at(ask_id);
            removeOrder(ask_loc);
        }
    }
}


// =============================================================================
// SECTION 8 — PUBLIC: cancelOrder()
// =============================================================================

bool OrderBook::cancelOrder(OrderId order_id) {
    auto map_it = order_map_.find(order_id);

    if (map_it == order_map_.end()) {
        return false;
    }

    const OrderLocation& loc = map_it->second;

    loc.it->status = OrderStatus::CANCELLED;

    removeOrder(loc);

    return true;
}


// =============================================================================
// SECTION 9 — PUBLIC: modifyOrder()
// =============================================================================

bool OrderBook::modifyOrder(OrderId order_id, Quantity new_qty) {
    auto map_it = order_map_.find(order_id);
    if (map_it == order_map_.end()) {
        return false;
    }

    OrderLocation& loc = map_it->second;
    Order& order = *loc.it;

    if (new_qty <= order.filled_quantity) {
        throw std::invalid_argument(
            "new_qty must exceed already filled quantity; "
            "use cancelOrder() to remove the order entirely.");
    }

    auto& target_map = (loc.side == Side::BID) ? bid_map_ : ask_map_;
    PriceLevel& level = target_map.at(loc.price);

    if (new_qty < order.quantity) {
        const Quantity old_remaining = order.remaining();
        order.quantity = new_qty;
        const Quantity new_remaining = order.remaining();

        level.total_qty -= (old_remaining - new_remaining);

    } else {
        Order replacement = *loc.it;
        replacement.quantity = new_qty;
        replacement.filled_quantity = order.filled_quantity;
        replacement.status = OrderStatus::OPEN;

        cancelOrder(order_id);
        addOrder(std::move(replacement));
    }

    return true;
}


// =============================================================================
// SECTION 10 — READ-ONLY ACCESSORS
// =============================================================================

std::optional<Price> OrderBook::bestBid() const noexcept {
    if (bid_map_.empty()) return std::nullopt;
    return bid_map_.rbegin()->first;
}

std::optional<Price> OrderBook::bestAsk() const noexcept {
    if (ask_map_.empty()) return std::nullopt;
    return ask_map_.begin()->first;
}

std::optional<double> OrderBook::midPrice() const noexcept {
    const auto bid = bestBid();
    const auto ask = bestAsk();
    if (!bid || !ask) return std::nullopt;
    return static_cast<double>(*bid + *ask) / 2.0;
}

std::optional<Price> OrderBook::spread() const noexcept {
    const auto bid = bestBid();
    const auto ask = bestAsk();
    if (!bid || !ask) return std::nullopt;
    return *ask - *bid;
}

Quantity OrderBook::volumeAtPrice(Side side, Price price) const noexcept {
    const auto& target_map = (side == Side::BID) ? bid_map_ : ask_map_;
    const auto it = target_map.find(price);
    if (it == target_map.end()) return Quantity{0};
    return it->second.total_qty;
}

const Order* OrderBook::findOrder(OrderId order_id) const noexcept {
    const auto it = order_map_.find(order_id);
    if (it == order_map_.end()) return nullptr;
    return &(*it->second.it);
}

std::size_t OrderBook::totalOrderCount() const noexcept {
    return active_orders_;
}

const std::map<Price, PriceLevel>& OrderBook::bidLadder() const noexcept {
    return bid_map_;
}

const std::map<Price, PriceLevel>& OrderBook::askLadder() const noexcept {
    return ask_map_;
}


// =============================================================================
// SECTION 10b — NEW: getTopVolumes()
// =============================================================================
//
//  DESIGN NOTES FOR RL INTEGRATION
//  ────────────────────────────────
//  An RL agent typically consumes the LOB state as a flat numeric feature
//  vector. The two key requirements are:
//
//    1. FIXED LENGTH: The observation space must be constant across every
//       time step. We guarantee this with zero-padding when the book is
//       shallow (fewer levels than requested depth).
//
//    2. BEST-FIRST ORDERING: The most informative (nearest-to-mid) level is
//       placed at index 0 so that convolutional or attention layers can learn
//       spatial relationships without an offset lookup.
//
//  TRAVERSAL STRATEGY:
//    ASK side: iterate FORWARD from begin() — ascending prices, so begin()
//              is the cheapest (best) ask. Each step moves one tick away.
//
//    BID side: iterate REVERSE from rbegin() — the sorted map is ascending,
//              so rbegin() is the highest (best) bid. Each step moves one
//              tick away from mid.
//
//  COMPLEXITY: O(depth) — we visit at most `depth` tree nodes, each in O(1)
//  since iterator increment on std::map is amortised O(1) (red-black tree
//  in-order successor walk).

std::vector<Quantity> OrderBook::getTopVolumes(Side side, int depth) const {
    // Allocate the result vector up-front and zero-fill it.
    // Any levels beyond what the book currently has will remain 0.
    std::vector<Quantity> result(static_cast<std::size_t>(depth), Quantity{0});

    if (depth <= 0) {
        return result;  // Edge case: caller asked for zero depth.
    }

    if (side == Side::ASK) {
        // ── ASK side: best ask = lowest price = map begin() ──────────────────
        //
        //  Walk forward through ask_map_ up to `depth` levels.
        //  begin() → 2nd cheapest → 3rd cheapest → ...
        //
        //  Operator++ on a std::map iterator is amortised O(1) (in-order
        //  tree successor). We stop at end() or when we've filled `depth`
        //  slots — whichever comes first.
        auto it = ask_map_.begin();
        for (int i = 0; i < depth && it != ask_map_.end(); ++i, ++it) {
            result[static_cast<std::size_t>(i)] = it->second.total_qty;
        }
    } else {
        // ── BID side: best bid = highest price = map rbegin() ────────────────
        //
        //  Walk backward through bid_map_ up to `depth` levels.
        //  rbegin() → 2nd highest → 3rd highest → ...
        //
        //  Operator++ on a reverse_iterator steps toward lower prices (i.e.,
        //  further from mid). We stop at rend() or when we've filled `depth`
        //  slots — whichever comes first.
        auto it = bid_map_.rbegin();
        for (int i = 0; i < depth && it != bid_map_.rend(); ++i, ++it) {
            result[static_cast<std::size_t>(i)] = it->second.total_qty;
        }
    }

    return result;  // Any unfilled tail slots remain 0 (from initialization).
}


// =============================================================================
// SECTION 11 — DIAGNOSTICS: assertInvariants()
// =============================================================================

bool OrderBook::assertInvariants() const {
    bool ok = true;
    std::size_t counted_orders = 0;

    auto check_side = [&](const std::map<Price, PriceLevel>& price_map,
                          Side expected_side,
                          const char* side_name) {
        for (const auto& [price, level] : price_map) {

            Quantity sum = 0;
            for (const Order& o : level.orders) {
                sum += o.remaining();
                ++counted_orders;

                if (o.status == OrderStatus::FILLED ||
                    o.status == OrderStatus::CANCELLED) {
                    std::cerr << "[INVARIANT FAIL] " << side_name
                              << " level " << price << " contains dead order "
                              << o.id << " with status "
                              << static_cast<int>(o.status) << "\n";
                    ok = false;
                }

                if (o.side != expected_side) {
                    std::cerr << "[INVARIANT FAIL] Order " << o.id
                              << " has wrong side in " << side_name << " map.\n";
                    ok = false;
                }

                if (order_map_.find(o.id) == order_map_.end()) {
                    std::cerr << "[INVARIANT FAIL] Order " << o.id
                              << " in list but missing from order_map_.\n";
                    ok = false;
                }
            }

            if (sum != level.total_qty) {
                std::cerr << "[INVARIANT FAIL] " << side_name
                          << " level " << price << " total_qty=" << level.total_qty
                          << " but sum of remaining()=" << sum << "\n";
                ok = false;
            }
        }
    };

    check_side(bid_map_, Side::BID, "BID");
    check_side(ask_map_, Side::ASK, "ASK");

    if (order_map_.size() != counted_orders) {
        std::cerr << "[INVARIANT FAIL] order_map_.size()=" << order_map_.size()
                  << " but counted " << counted_orders << " orders in lists.\n";
        ok = false;
    }

    if (active_orders_ != counted_orders) {
        std::cerr << "[INVARIANT FAIL] active_orders_=" << active_orders_
                  << " but counted " << counted_orders << " orders in lists.\n";
        ok = false;
    }

    return ok;
}


// =============================================================================
// SECTION 12 — MatchingEngine (multi-instrument orchestration)
// =============================================================================

MatchingEngine::MatchingEngine(TradeCallback global_trade_cb)
    : global_trade_cb_(std::move(global_trade_cb))
{
    books_.reserve(64);
}

void MatchingEngine::addInstrument(const std::string& instrument_id) {
    books_.emplace(instrument_id,
                   std::make_unique<OrderBook>(instrument_id, global_trade_cb_));
}

OrderId MatchingEngine::submitOrder(const std::string& instrument_id, Order order) {
    auto it = books_.find(instrument_id);
    if (it == books_.end()) {
        throw std::out_of_range("Instrument not registered: " + instrument_id);
    }
    return it->second->addOrder(std::move(order));
}

bool MatchingEngine::cancelOrder(const std::string& instrument_id, OrderId order_id) {
    auto it = books_.find(instrument_id);
    if (it == books_.end()) return false;
    return it->second->cancelOrder(order_id);
}

const OrderBook* MatchingEngine::getBook(const std::string& instrument_id) const noexcept {
    const auto it = books_.find(instrument_id);
    if (it == books_.end()) return nullptr;
    return it->second.get();
}

} // namespace lob
