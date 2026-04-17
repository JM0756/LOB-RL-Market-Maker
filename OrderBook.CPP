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
//
//  This function is the "write path" into the book's two data structures.
//  It is called by addOrder() whenever an order is not immediately filled
//  (or only partially filled) and must rest in the book.
//
//  THE TWO-STRUCTURE WRITE:
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │  Structure A:  bid_map_ / ask_map_   (std::map — the sorted tree)   │
//  │                                                                      │
//  │    map[price] creates a new PriceLevel if none exists, or returns   │
//  │    a reference to the existing one. Either way: O(log P).           │
//  │    We then push the order onto the BACK of the level's list: O(1).  │
//  │                                                                      │
//  │  Structure B:  order_map_            (unordered_map — the index)    │
//  │                                                                      │
//  │    We store the iterator (pointer) to the node we just pushed.      │
//  │    This is the key insight: std::list guarantees that iterators to   │
//  │    existing nodes remain valid even after insertions/erasures        │
//  │    elsewhere in the list. So this pointer is stable for the order's  │
//  │    entire lifetime. Insert: O(1) amortised.                          │
//  └──────────────────────────────────────────────────────────────────────┘

void OrderBook::insertResting(Order&& order) {
    // Determine which price ladder to insert into.
    auto& target_map = (order.side == Side::BID) ? bid_map_ : ask_map_;

    // --- STRUCTURE A: Sorted tree insertion ---
    //
    //  map::operator[] does one of two things:
    //    (a) Price level already exists → O(log P) tree lookup, return ref.
    //    (b) Price level is new         → O(log P) tree insert + default-construct
    //        a fresh PriceLevel. We then initialise it below.
    //
    //  Either way, `level` is a reference to the canonical PriceLevel object
    //  that lives inside the tree node. It will not move in memory.
    PriceLevel& level = target_map[order.price];

    // If this is a newly created level, initialise its price field.
    // (total_qty starts at 0 from default construction.)
    if (level.orders.empty()) {
        level.price = order.price;
    }

    // Accumulate this order's quantity into the level's visible aggregate.
    // This is the value shown to market data consumers as the level's "size".
    level.total_qty += order.remaining();

    // Push the order onto the BACK of the FIFO queue.
    //  - Back = lowest time priority (most recently arrived).
    //  - Front = highest time priority (oldest, matched first).
    // push_back() returns void, so we use emplace_back and grab the iterator
    // via std::prev(end()) after insertion.
    level.orders.push_back(std::move(order));

    // Grab a stable iterator pointing directly at the node we just inserted.
    // std::prev(end()) is O(1) for std::list.
    OrderIterator it = std::prev(level.orders.end());

    // --- STRUCTURE B: Hash map registration ---
    //
    //  We record: { which side, which price, which list node }.
    //  This is the lookup entry that makes O(1) cancellation possible.
    //  The iterator `it` is our direct "pointer" into the list — no scan needed.
    //
    //  O(1) amortised: hash insertion, no collision assumed in the normal path.
    order_map_.emplace(it->id, OrderLocation{ it->side, it->price, it });

    ++active_orders_;
}


// =============================================================================
// SECTION 3 — PRIVATE PRIMITIVE: removeOrder()
// =============================================================================
//
//  This is the "delete path" shared by BOTH matching (filled orders) and
//  cancellation (cancelled orders). It accepts an OrderLocation — the struct
//  we stored in order_map_ — which bundles everything we need.
//
//  THE O(1) CANCELLATION MECHANISM — STEP BY STEP:
//
//  Given OrderLocation { side=BID, price=15000, it=<list iterator> }:
//
//  Step 1 [O(1)]:   it->remaining() gives us the quantity to subtract.
//  Step 2 [O(1)]:   level.orders.erase(it)
//                   ─── The doubly-linked list relinks two pointers:
//                       prev->next = it->next
//                       next->prev = it->prev
//                   ─── No element shifting. No memory copying. Just
//                       two pointer assignments regardless of list size.
//  Step 3 [O(log P)]: If the level is now empty, erase it from the tree.
//                   ─── This is O(log P) because the red-black tree must
//                       rebalance. This is the ONLY non-O(1) step.
//  Step 4 [O(1)]:   order_map_.erase(id) removes the index entry.

void OrderBook::removeOrder(const OrderLocation& loc) {
    // Resolve which map this order lives in.
    auto& target_map = (loc.side == Side::BID) ? bid_map_ : ask_map_;

    // Locate the PriceLevel in the tree.
    // find() is O(log P). We must use find(), not operator[], because we don't
    // want to accidentally default-construct a new level if something goes wrong.
    auto map_it = target_map.find(loc.price);

    // Defensive check: in a correct implementation this should never fire.
    // The level must exist if the order_map_ entry for it exists.
    assert(map_it != target_map.end() && "OrderLocation points to nonexistent price level");

    PriceLevel& level = map_it->second;

    // --- STEP 1: Subtract this order's remaining quantity from the level total.
    //  O(1): simple integer subtraction.
    //  We use remaining() because filled_quantity may have been partially reduced
    //  by a prior partial fill before this removal (e.g., the order was PARTIALLY
    //  FILLED and now we are removing the remainder after a full fill).
    level.total_qty -= loc.it->remaining();

    // --- STEP 2: THE O(1) LIST NODE ERASURE ---
    //
    //  This is the payoff of storing the iterator in order_map_.
    //  std::list::erase(iterator) is O(1) — it just relinks neighbours.
    //
    //  Memory layout before erase:
    //    [prev_node] <──> [THIS NODE (loc.it)] <──> [next_node]
    //
    //  Memory layout after erase:
    //    [prev_node] <──> [next_node]
    //              (THIS NODE freed)
    //
    //  No scanning, no shifting, no searching. This is why we use std::list.
    const OrderId erased_id = loc.it->id;   // Save ID before node is destroyed.
    level.orders.erase(loc.it);             // ← THE O(1) OPERATION

    // --- STEP 3: Prune the price level from the tree if it is now empty.
    //
    //  An empty PriceLevel is meaningless — it would pollute the spread
    //  calculation and mislead market data consumers.
    //
    //  map::erase(iterator) is O(log P) because the red-black tree must
    //  potentially rebalance (rotate/recolour nodes) to maintain its invariants.
    //  This is the single unavoidable O(log P) cost in the cancellation path.
    if (level.empty()) {
        target_map.erase(map_it);   // ← O(log P), unavoidable
    }

    // --- STEP 4: Remove the index entry from the hash map.
    //  O(1) amortised: single hash bucket lookup and erasure.
    order_map_.erase(erased_id);    // ← O(1)

    --active_orders_;
}


// =============================================================================
// SECTION 4 — PRIVATE PRIMITIVE: executeFill()
// =============================================================================
//
//  Records a single "fill event" — a quantum of a match between one bid order
//  and one ask order. May be called multiple times for a single order if it
//  sweeps through several price levels or several orders at the same level.
//
//  This function ONLY updates quantities and emits the callback.
//  It does NOT remove filled orders from the book — that is the caller's
//  (matchOrders') responsibility, because matchOrders needs to inspect the
//  order's state after the fill to decide whether to remove it.

void OrderBook::executeFill(Order& bid_order, Order& ask_order,
                            Price exec_price, Quantity exec_qty) {
    // Sanity: we should never attempt a zero-quantity fill.
    assert(exec_qty > 0);
    assert(exec_qty <= bid_order.remaining());
    assert(exec_qty <= ask_order.remaining());

    // --- Update the BID order ---
    bid_order.filled_quantity += exec_qty;
    bid_order.status = (bid_order.remaining() == 0)
                       ? OrderStatus::FILLED
                       : OrderStatus::PARTIALLY_FILLED;

    // --- Update the ASK order ---
    ask_order.filled_quantity += exec_qty;
    ask_order.status = (ask_order.remaining() == 0)
                       ? OrderStatus::FILLED
                       : OrderStatus::PARTIALLY_FILLED;

    // --- Emit the trade event ---
    //
    //  Build an immutable Trade record and fire the callback (if registered).
    //  In a production system this would publish to:
    //    - The exchange's consolidated trade tape.
    //    - The risk engine (for real-time P&L and margin).
    //    - Order management systems of both counterparties.
    //  We use steady_clock::now() for the execution timestamp.
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
//
//  Price-Time Priority (also called FIFO priority) means:
//    1. The best-priced order matches first.
//    2. Among orders at the same price, the earliest-submitted matches first.
//
//  For the EXECUTION PRICE specifically, the exchange convention is:
//    "The resting (passive) order's price determines the fill price."
//
//  This ensures that the resting order gets the price it was promised, and
//  the aggressive order benefits from any price improvement.
//
//  Example:
//    Resting ASK at $100.00 (passive)
//    Incoming BID at $101.00 (aggressive — crosses the spread)
//    Execution price = $100.00  (resting ask's price, not the aggressive bid's)
//    → The buyer gets price improvement: pays $100 instead of $101.

/*static*/
Price OrderBook::resolveExecPrice(const Order& resting,
                                  const Order& /*aggressor*/) noexcept {
    // The passive (resting) order's price wins.
    // The aggressor parameter is available for alternative pricing models
    // (e.g., midpoint matching used by some dark pools), hence the signature,
    // but for standard continuous double-auction we use the resting price.
    return resting.price;
}


// =============================================================================
// SECTION 6 — PUBLIC: addOrder()
// =============================================================================
//
//  The primary public entry point. Handles the full lifecycle of a new order:
//    1. Validate and stamp the order (ID, timestamp).
//    2. For MARKET orders: convert to an extreme-price LIMIT order.
//    3. Call matchOrders() to attempt immediate execution.
//    4. If quantity remains unfilled → insertResting() into the book.
//
//  WHY matchOrders() IS CALLED BEFORE insertResting():
//    An incoming aggressive order should be matched against existing resting
//    orders BEFORE it is placed in the book. If we inserted first, the order
//    would immediately match against itself in degenerate edge cases, and more
//    importantly, we'd need to handle the awkward state of an order being
//    simultaneously in the book and being matched.
//
//  OVERALL COMPLEXITY: O(K log P) where K = fills generated.
//  If the order does not match at all: O(log P) for the final insertResting().

OrderId OrderBook::addOrder(Order order) {
    // --- Validation ---
    if (order.quantity == 0) {
        throw std::invalid_argument("Order quantity must be greater than zero.");
    }

    // --- Stamp the order with server-assigned metadata ---
    //
    //  Overwrite any client-supplied ID and timestamp to prevent spoofing.
    //  IDs are monotonically increasing — this also provides a natural
    //  tie-breaker when two orders arrive at the same timestamp.
    order.id               = next_order_id_++;
    order.timestamp        = std::chrono::steady_clock::now();
    order.filled_quantity  = 0;
    order.status           = OrderStatus::OPEN;

    const OrderId assigned_id = order.id;   // Save before we move the order.

    // --- Market Order: Assign sentinel price ---
    //
    //  We model market orders as limit orders with an extreme price that
    //  guarantees they will cross any available spread:
    //    - Market BUY:  price = INT64_MAX → will beat any ask price.
    //    - Market SELL: price = 0         → will undercut any bid price.
    //
    //  This is the standard trick used by real matching engines (e.g., NYSE,
    //  LSE) to avoid a separate code path for market orders in matchOrders().
    if (order.type == OrderType::MARKET) {
        order.price = (order.side == Side::BID)
                      ? std::numeric_limits<Price>::max()
                      : Price{0};
    }

    // --- Attempt immediate matching ---
    //
    //  We temporarily insert the order into the book so that matchOrders() can
    //  see it as either the aggressor or a resting order, then let the matching
    //  loop handle it. However, there's a cleaner pattern: insert the order
    //  and let matchOrders sweep.
    //
    //  Implementation pattern used here: insert first, then match.
    //  matchOrders() will consume as much of this order as possible.
    //  Any remainder stays in the book as a resting order.
    insertResting(std::move(order));

    // matchOrders() inspects the current best bid and best ask.
    // If this new order is aggressive (i.e., it crosses the spread), the loop
    // will fire and consume it (fully or partially) against resting orders.
    matchOrders();

    return assigned_id;
}


// =============================================================================
// SECTION 7 — PUBLIC: matchOrders()
// =============================================================================
//
//  The core matching loop — the heart of the exchange engine.
//
//  ALGORITHM: Continuous Double Auction with Price-Time Priority
//  ─────────────────────────────────────────────────────────────
//
//  We repeatedly inspect the BEST BID and BEST ASK. A match is possible
//  when these two prices cross (bid >= ask). We execute fills until either:
//    (a) The spread opens back up (bid < ask), OR
//    (b) One side of the book is completely exhausted.
//
//  PRICE LADDER TRAVERSAL:
//
//    bid_map_ is a std::map<Price, PriceLevel> sorted ASCENDING by price.
//    ┌──────────────────────────────────────────────────────┐
//    │  bid_map_:  100 | 101 | 102 | 103 | 104 | 105       │
//    │                                         ↑ rbegin()  │
//    │                                   Best bid = 105     │
//    └──────────────────────────────────────────────────────┘
//
//    ask_map_ is also sorted ASCENDING.
//    ┌──────────────────────────────────────────────────────┐
//    │  ask_map_:  103 | 104 | 106 | 108 | 110             │
//    │              ↑ begin()                               │
//    │         Best ask = 103                               │
//    └──────────────────────────────────────────────────────┘
//
//    Best bid (105) >= Best ask (103) → MATCH at 103 (resting ask's price).
//
//  FIFO ORDER WITHIN A PRICE LEVEL:
//
//    When the matching loop arrives at a price level, it always consumes
//    from the FRONT of the list — the oldest (earliest submitted) order.
//    This enforces time priority within the price level.
//
//  PARTIAL FILL EXAMPLE:
//
//    Resting ASK: id=1, qty=100 at $103
//    Incoming BID: id=2, qty=200 at $105
//
//    Pass 1:  exec_qty = min(200, 100) = 100
//             ASK id=1: filled_qty=100, status=FILLED   → remove from book
//             BID id=2: filled_qty=100, status=PARTIALLY_FILLED, remaining=100
//
//    No more asks at $103 or below $105 → spread opens → loop exits.
//    BID id=2 stays in book at $105 with remaining qty = 100.
//
//  COMPLEXITY: O(K log P) where K = total fills across all price levels.
//  Each fill: O(1) list front-pop + O(log P) potential level erasure.

void OrderBook::matchOrders() {
    // Loop until no crossing exists between the best bid and best ask.
    while (!bid_map_.empty() && !ask_map_.empty()) {

        // --- PEEK AT THE BEST BID ---
        //
        //  bid_map_ is sorted ascending, so the HIGHEST price (best bid) is
        //  at the rightmost node: rbegin().
        //
        //  rbegin() returns a reverse_iterator. Dereferencing gives a
        //  std::pair<const Price, PriceLevel>. We work with the PriceLevel.
        //
        //  O(1): rbegin() on std::map is a cached pointer to the max node.
        auto best_bid_it = bid_map_.rbegin();       // Reverse iterator to max price.
        PriceLevel& bid_level = best_bid_it->second;

        // --- PEEK AT THE BEST ASK ---
        //
        //  ask_map_ sorted ascending → lowest price (best ask) is at begin().
        //  O(1): begin() is always a cached pointer to the min node.
        auto best_ask_it = ask_map_.begin();        // Forward iterator to min price.
        PriceLevel& ask_level = best_ask_it->second;

        // --- CHECK FOR PRICE CROSSING ---
        //
        //  A match is only possible when bid.price >= ask.price.
        //  If the spread is positive (bid < ask), there is nothing to match.
        if (bid_level.price < ask_level.price) {
            break;  // No crossing. Book is in a valid resting state.
        }

        // ─────────────────────────────────────────────────────────────────────
        //  A MATCH EXISTS. Process the front orders of each level (FIFO).
        // ─────────────────────────────────────────────────────────────────────

        // The front of the bid level's list is the OLDEST bid at this price.
        // The front of the ask level's list is the OLDEST ask at this price.
        // "Oldest" = highest time priority = matched first.
        //
        //  O(1): list::front() returns a reference to the head node.
        Order& bid_order = bid_level.orders.front();
        Order& ask_order = ask_level.orders.front();

        // --- DETERMINE EXECUTION PRICE ---
        //
        //  Which order arrived first? The one with the lower ID (IDs are
        //  monotonically assigned by the server, so lower ID = earlier arrival).
        //  The earlier order is the "resting" (passive) side; its price wins.
        //
        //  This handles the edge case where both orders were just inserted
        //  within the same matching cycle.
        const bool bid_is_older   = (bid_order.id < ask_order.id);
        const Order& resting_order  = bid_is_older ? bid_order : ask_order;
        const Order& aggressing_order = bid_is_older ? ask_order : bid_order;

        const Price exec_price = resolveExecPrice(resting_order, aggressing_order);

        // --- DETERMINE EXECUTION QUANTITY ---
        //
        //  The fill size is constrained by whichever order has LESS remaining qty.
        //  This is the partial fill logic.
        //
        //  Case A: bid.remaining == ask.remaining  → both fully filled.
        //  Case B: bid.remaining  < ask.remaining  → bid fully filled,
        //                                             ask partially filled.
        //  Case C: bid.remaining  > ask.remaining  → ask fully filled,
        //                                             bid partially filled.
        //
        //  In all cases, exec_qty = min(bid.remaining, ask.remaining).
        //  std::min is O(1).
        const Quantity exec_qty = std::min(bid_order.remaining(),
                                           ask_order.remaining());

        // --- EXECUTE THE FILL ---
        //
        //  executeFill() updates filled_quantity and status on both orders,
        //  and fires the trade callback. It does NOT remove orders from the book.
        //
        //  We must update order state BEFORE we touch the price levels' total_qty,
        //  because removeOrder() reads remaining() — which depends on filled_quantity.
        executeFill(bid_order, ask_order, exec_price, exec_qty);

        // --- UPDATE PRICE LEVEL AGGREGATES ---
        //
        //  Decrement total_qty on both levels immediately.
        //  This keeps the Level 2 market data view consistent in real-time,
        //  even for orders that are only partially filled (they stay in the book
        //  but show reduced volume).
        bid_level.total_qty -= exec_qty;
        ask_level.total_qty -= exec_qty;

        // ─────────────────────────────────────────────────────────────────────
        //  REMOVAL OF FULLY-FILLED ORDERS
        //
        //  An order is fully filled when remaining() == 0 (i.e., status == FILLED).
        //  We remove it from BOTH the list and the hash map index.
        //
        //  We must check bid and ask independently — in Case A above, BOTH
        //  orders are filled in the same fill event and both need removal.
        //  In Cases B and C, only one is fully filled.
        //
        //  NOTE: We look up the OrderLocation from order_map_ to get the
        //  iterator — we DO NOT use the reference `bid_order` / `ask_order`
        //  after erasing because erasing the list node invalidates the reference.
        //  We therefore check the status BEFORE calling removeOrder.
        // ─────────────────────────────────────────────────────────────────────

        const bool bid_filled = (bid_order.status == OrderStatus::FILLED);
        const bool ask_filled = (ask_order.status == OrderStatus::FILLED);

        // Save IDs before any potential erasure invalidates references.
        const OrderId bid_id = bid_order.id;
        const OrderId ask_id = ask_order.id;

        if (bid_filled) {
            // --- REMOVE FILLED BID ---
            //
            //  1. Look up the OrderLocation from the hash map.   O(1)
            //  2. list::erase(iterator) unlinks the node.        O(1) ← THE KEY LINE
            //  3. If the price level is now empty, erase it
            //     from bid_map_.                                  O(log P)
            //  4. Erase the entry from order_map_.               O(1)
            //
            //  Total: O(log P) dominated by the tree level cleanup.
            //
            //  IMPORTANT: After removeOrder(), the `bid_level` reference and
            //  `best_bid_it` may be dangling if the level was erased. We must
            //  NOT dereference them after this point in this iteration.
            const OrderLocation& bid_loc = order_map_.at(bid_id);
            removeOrder(bid_loc);
            // `bid_level`, `bid_order`, `best_bid_it` are now potentially dangling.
        }

        if (ask_filled) {
            // --- REMOVE FILLED ASK ---
            //  Same O(1)+O(log P) sequence as the bid removal above.
            const OrderLocation& ask_loc = order_map_.at(ask_id);
            removeOrder(ask_loc);
            // `ask_level`, `ask_order`, `best_ask_it` are now potentially dangling.
        }

        // Loop back: re-evaluate the spread with the updated book state.
        // This handles the case where multiple orders at different price levels
        // need to be swept (e.g., a large market order eating through 5 levels).
    }
}


// =============================================================================
// SECTION 8 — PUBLIC: cancelOrder()
// =============================================================================
//
//  Cancels a resting order by its ID. This is the primary use case for the
//  order_map_ hash map — without it, we would have to scan the price level's
//  list to find the order, turning an O(1) operation into O(N/P).
//
//  FULL CANCELLATION TRACE (for id = 42, a BID at price $105):
//
//    order_map_.find(42)
//      → returns OrderLocation { side=BID, price=10500, it=<ptr to list node> }
//
//    bid_map_.find(10500)
//      → O(log P): returns PriceLevel& for $105
//
//    level.orders.erase(it)
//      → O(1): THE O(1) POINTER RELINK. No other nodes are touched.
//
//    if (level.empty()) bid_map_.erase(...)
//      → O(log P): tree rebalance, only if level is now empty
//
//    order_map_.erase(42)
//      → O(1): hash bucket deletion
//
//  NET: O(log P) — the tree erase is the bottleneck, list erase is O(1).

bool OrderBook::cancelOrder(OrderId order_id) {
    // --- Step 1: Hash map lookup ---
    //
    //  find() is used (not operator[]) to avoid creating a spurious entry
    //  if the order_id does not exist.
    //  O(1) average, O(N) worst-case (hash collision — negligible in practice).
    auto map_it = order_map_.find(order_id);

    if (map_it == order_map_.end()) {
        // Order not found — it may have already been filled or never existed.
        return false;
    }

    const OrderLocation& loc = map_it->second;

    // Mark the order as cancelled before removal for any observers that may
    // inspect the order object's state before the node is destroyed.
    loc.it->status = OrderStatus::CANCELLED;

    // --- Step 2: Delegate to removeOrder() ---
    //
    //  removeOrder() performs the O(1) list erasure and the O(log P) tree cleanup.
    //  After this call, the list node is destroyed, `loc` is dangling,
    //  and order_map_[order_id] no longer exists.
    removeOrder(loc);   // ← Contains the O(1) list::erase and O(log P) tree cleanup.

    return true;
}


// =============================================================================
// SECTION 9 — PUBLIC: modifyOrder()
// =============================================================================
//
//  Modifies the quantity of a resting order.
//
//  REDUCING quantity is an in-place operation — no re-queuing needed.
//  The order retains its time priority (queue position) in the level's list.
//
//  INCREASING quantity would require cancelling and resubmitting, losing
//  queue position. This is enforced by the function's contract.

bool OrderBook::modifyOrder(OrderId order_id, Quantity new_qty) {
    auto map_it = order_map_.find(order_id);
    if (map_it == order_map_.end()) {
        return false;   // Order not found.
    }

    OrderLocation& loc = map_it->second;
    Order& order = *loc.it;

    if (new_qty <= order.filled_quantity) {
        throw std::invalid_argument(
            "new_qty must exceed already filled quantity; "
            "use cancelOrder() to remove the order entirely.");
    }

    // Which map are we operating on?
    auto& target_map = (loc.side == Side::BID) ? bid_map_ : ask_map_;
    PriceLevel& level = target_map.at(loc.price);

    if (new_qty < order.quantity) {
        // --- REDUCING: In-place update. O(1). ---
        //
        //  Adjust the level's aggregate by the delta, then update the order.
        //  The order stays in its current list position — time priority preserved.
        const Quantity old_remaining = order.remaining();
        order.quantity = new_qty;
        const Quantity new_remaining = order.remaining();

        level.total_qty -= (old_remaining - new_remaining);    // O(1)

    } else {
        // --- INCREASING: Cancel and resubmit. O(log P) + O(log P). ---
        //
        //  We must re-queue to maintain fairness (this order goes to the back).
        //  Build the replacement order, cancel the old one, then add the new one.
        Order replacement = *loc.it;    // Copy before erasure.
        replacement.quantity = new_qty;
        replacement.filled_quantity = order.filled_quantity;
        replacement.status = OrderStatus::OPEN;
        // Timestamp will be refreshed by addOrder(), correctly moving it to the
        // back of the queue.

        cancelOrder(order_id);              // Removes from all structures. O(log P)
        addOrder(std::move(replacement));   // Re-inserts and re-matches.  O(K log P)
    }

    return true;
}


// =============================================================================
// SECTION 10 — READ-ONLY ACCESSORS
// =============================================================================

std::optional<Price> OrderBook::bestBid() const noexcept {
    if (bid_map_.empty()) return std::nullopt;
    // rbegin() is the iterator to the highest-priced level. O(1).
    return bid_map_.rbegin()->first;
}

std::optional<Price> OrderBook::bestAsk() const noexcept {
    if (ask_map_.empty()) return std::nullopt;
    // begin() is the iterator to the lowest-priced level. O(1).
    return ask_map_.begin()->first;
}

std::optional<double> OrderBook::midPrice() const noexcept {
    const auto bid = bestBid();
    const auto ask = bestAsk();
    if (!bid || !ask) return std::nullopt;
    // Divide by 2.0 (double) to avoid integer truncation.
    return static_cast<double>(*bid + *ask) / 2.0;
}

std::optional<Price> OrderBook::spread() const noexcept {
    const auto bid = bestBid();
    const auto ask = bestAsk();
    if (!bid || !ask) return std::nullopt;
    return *ask - *bid;     // Positive when book is in a valid resting state.
}

Quantity OrderBook::volumeAtPrice(Side side, Price price) const noexcept {
    // O(log P): single tree lookup.
    const auto& target_map = (side == Side::BID) ? bid_map_ : ask_map_;
    const auto it = target_map.find(price);
    if (it == target_map.end()) return Quantity{0};
    return it->second.total_qty;
}

const Order* OrderBook::findOrder(OrderId order_id) const noexcept {
    // O(1) average: hash map lookup.
    const auto it = order_map_.find(order_id);
    if (it == order_map_.end()) return nullptr;
    return &(*it->second.it);   // Dereference the stored list iterator.
}

std::size_t OrderBook::totalOrderCount() const noexcept {
    return active_orders_;      // O(1): cached counter.
}

const std::map<Price, PriceLevel>& OrderBook::bidLadder() const noexcept {
    return bid_map_;
}

const std::map<Price, PriceLevel>& OrderBook::askLadder() const noexcept {
    return ask_map_;
}


// =============================================================================
// SECTION 11 — DIAGNOSTICS: assertInvariants()
// =============================================================================
//
//  A full O(N) consistency sweep. Use only in debug builds or test harnesses.
//
//  INVARIANTS CHECKED:
//    I1: order_map_.size() == total orders across all price level lists.
//    I2: PriceLevel::total_qty == sum of remaining() for all orders in it.
//    I3: No order in any list has status FILLED or CANCELLED.
//    I4: Every order's side matches which map it is in (bid vs ask).
//    I5: Every order in a list is reachable from order_map_ by its ID.

bool OrderBook::assertInvariants() const {
    bool ok = true;
    std::size_t counted_orders = 0;

    // Helper lambda to scan one side of the book.
    auto check_side = [&](const std::map<Price, PriceLevel>& price_map,
                          Side expected_side,
                          const char* side_name) {
        for (const auto& [price, level] : price_map) {

            // I2: Validate total_qty aggregate.
            Quantity sum = 0;
            for (const Order& o : level.orders) {
                sum += o.remaining();
                ++counted_orders;

                // I3: No dead orders should remain in any list.
                if (o.status == OrderStatus::FILLED ||
                    o.status == OrderStatus::CANCELLED) {
                    std::cerr << "[INVARIANT FAIL] " << side_name
                              << " level " << price << " contains dead order "
                              << o.id << " with status "
                              << static_cast<int>(o.status) << "\n";
                    ok = false;
                }

                // I4: Side field matches map side.
                if (o.side != expected_side) {
                    std::cerr << "[INVARIANT FAIL] Order " << o.id
                              << " has wrong side in " << side_name << " map.\n";
                    ok = false;
                }

                // I5: Order is reachable from order_map_.
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

    // I1: Hash map size must match total list entries.
    if (order_map_.size() != counted_orders) {
        std::cerr << "[INVARIANT FAIL] order_map_.size()=" << order_map_.size()
                  << " but counted " << counted_orders << " orders in lists.\n";
        ok = false;
    }

    // active_orders_ counter must also match.
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
    books_.reserve(64);  // Reasonable default for a small-to-medium venue.
}

void MatchingEngine::addInstrument(const std::string& instrument_id) {
    // emplace avoids an extra move/copy of the unique_ptr.
    // If the instrument already exists, this is a no-op (emplace does not
    // overwrite existing keys).
    books_.emplace(instrument_id,
                   std::make_unique<OrderBook>(instrument_id, global_trade_cb_));
}

OrderId MatchingEngine::submitOrder(const std::string& instrument_id, Order order) {
    // O(1) avg: hash map lookup for the instrument.
    auto it = books_.find(instrument_id);
    if (it == books_.end()) {
        throw std::out_of_range("Instrument not registered: " + instrument_id);
    }
    // Delegate entirely to the per-instrument OrderBook.
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