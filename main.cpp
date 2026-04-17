#include "OrderBook.H"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string sideToString(lob::Side side) {
    return (side == lob::Side::BID) ? "BID" : "ASK";
}

std::string typeToString(lob::OrderType type) {
    return (type == lob::OrderType::MARKET) ? "MARKET" : "LIMIT";
}

std::string formatPrice(lob::Price price_in_cents) {
    std::ostringstream out;
    out << '$' << std::fixed << std::setprecision(2)
        << (static_cast<double>(price_in_cents) / 100.0);
    return out.str();
}

lob::Order makeLimitOrder(lob::Side side, lob::Price price, lob::Quantity qty) {
    return lob::Order {
        0,
        price,
        qty,
        0,
        std::chrono::steady_clock::time_point {},
        side,
        lob::OrderType::LIMIT,
        lob::OrderStatus::OPEN
    };
}

lob::Order makeMarketOrder(lob::Side side, lob::Quantity qty) {
    return lob::Order {
        0,
        0,
        qty,
        0,
        std::chrono::steady_clock::time_point {},
        side,
        lob::OrderType::MARKET,
        lob::OrderStatus::OPEN
    };
}

void printOneSide(const std::map<lob::Price, lob::PriceLevel>& ladder, lob::Side side) {
    const char* title = (side == lob::Side::ASK) ? "ASKS (best -> worst)" : "BIDS (best -> worst)";
    std::cout << title << '\n';

    if (ladder.empty()) {
        std::cout << "  [none]\n";
        return;
    }

    if (side == lob::Side::ASK) {
        for (const auto& [price, level] : ladder) {
            std::cout << "  " << formatPrice(price)
                      << "  total=" << level.total_qty
                      << "  orders=";
            bool first = true;
            for (const auto& order : level.orders) {
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << "id " << order.id << " qty " << order.remaining();
                first = false;
            }
            std::cout << '\n';
        }
    } else {
        for (auto it = ladder.rbegin(); it != ladder.rend(); ++it) {
            const auto& price = it->first;
            const auto& level = it->second;
            std::cout << "  " << formatPrice(price)
                      << "  total=" << level.total_qty
                      << "  orders=";
            bool first = true;
            for (const auto& order : level.orders) {
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << "id " << order.id << " qty " << order.remaining();
                first = false;
            }
            std::cout << '\n';
        }
    }
}

void printBookSnapshot(const lob::OrderBook& book, const std::string& heading) {
    std::cout << "\n=== " << heading << " ===\n";
    printOneSide(book.askLadder(), lob::Side::ASK);
    printOneSide(book.bidLadder(), lob::Side::BID);

    const auto best_bid = book.bestBid();
    const auto best_ask = book.bestAsk();

    std::cout << "Best Bid: " << (best_bid ? formatPrice(*best_bid) : "N/A") << '\n';
    std::cout << "Best Ask: " << (best_ask ? formatPrice(*best_ask) : "N/A") << '\n';
    std::cout << "Open Orders: " << book.totalOrderCount() << '\n';
}

}  // namespace

int main() {
    std::cout << "=== Limit Order Book Demo ===\n";

    lob::OrderBook book(
        "AAPL",
        [](const lob::Trade& trade) {
            std::cout << "Trade Executed: "
                      << trade.exec_quantity << " shares @ "
                      << formatPrice(trade.exec_price)
                      << " (bid_id=" << trade.bid_order_id
                      << ", ask_id=" << trade.ask_order_id << ")\n";
        });

    auto submit_order = [&](lob::Order order) {
        const lob::Side side = order.side;
        const lob::OrderType type = order.type;
        const lob::Quantity qty = order.quantity;
        const lob::Price price = order.price;

        const lob::OrderId id = book.addOrder(std::move(order));

        std::cout << "Order Added: "
                  << sideToString(side) << ' '
                  << typeToString(type)
                  << "  id=" << id
                  << "  qty=" << qty;

        if (type == lob::OrderType::LIMIT) {
            std::cout << "  @ " << formatPrice(price);
        }

        std::cout << '\n';
        return id;
    };

    std::cout << "\nBuilding initial non-crossed book...\n";

    submit_order(makeLimitOrder(lob::Side::BID, 15000, 120));
    submit_order(makeLimitOrder(lob::Side::BID, 14995, 80));
    submit_order(makeLimitOrder(lob::Side::BID, 14990, 140));

    submit_order(makeLimitOrder(lob::Side::ASK, 15005, 70));
    submit_order(makeLimitOrder(lob::Side::ASK, 15010, 90));
    submit_order(makeLimitOrder(lob::Side::ASK, 15015, 120));

    printBookSnapshot(book, "Book After Initial Build");

    std::cout << "\nSubmitting aggressive incoming order...\n";
    submit_order(makeMarketOrder(lob::Side::BID, 200));

    printBookSnapshot(book, "Final Book (Remaining Open Orders)");

    if (!book.assertInvariants()) {
        std::cerr << "Invariant check failed.\n";
        return 1;
    }

    std::cout << "\nInvariant check passed.\n";
    return 0;
}