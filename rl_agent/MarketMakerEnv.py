"""
MarketMakerEnv.py
─────────────────
A custom Gymnasium environment that wraps the `lob_engine` C++ Limit Order
Book as the market simulator for training a Reinforcement Learning market-
making agent.

ENVIRONMENT OVERVIEW
────────────────────
The agent acts as a market maker on a single instrument. Each step it chooses
one of four quoting strategies (Hold, Tight, Wide, Clear). A synthetic market
flow process generates random market orders each step, which may fill the
agent's resting quotes. The agent is rewarded for capturing the bid-ask spread
(P&L) but penalised for carrying inventory (risk).

REWARD FORMULA
──────────────
    reward = Δ(TotalWealth) - λ * inventory²

    TotalWealth  = cash + inventory × mid_price   (mark-to-market)
    λ            = inventory penalty coefficient (default 0.01)

The λ term is the Avellaneda-Stoikov inventory penalty — it creates a
restoring force that pushes the agent to stay flat, mimicking a risk-averse
market maker who tightens/widens quotes based on signed inventory.

MARKET SIMULATION
─────────────────
There are no real counterparties. Each step, the environment:
  1. Seeds background resting limit orders around the current mid-price to
     maintain a realistic-looking Level 2 ladder.
  2. Generates a random market order (buy or sell) drawn from a Poisson
     arrival process. This simulates uninformed order flow hitting the book.

The agent's own resting quotes sit inside the same lob_engine book and are
filled whenever the simulated flow crosses them — exactly as they would be on
a real exchange.

PRICE CONVENTION
────────────────
All prices inside lob_engine are fixed-point integers (cents).
Example:  $150.00 → 15000

Observations and rewards are kept in cent units throughout to avoid
floating-point rescaling in the inner loop. Normalise in your policy wrapper
if needed (e.g., divide mid-price by initial_mid to get a unit-scale input).

DEPENDENCIES
────────────
    pip install gymnasium numpy
    # lob_engine must be compiled and importable — see CMakeLists.txt
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import lob_engine as lob


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Number of price levels exposed in the observation vector.
LOB_DEPTH: int = 3

# Fixed-point tick size in the engine (1 cent = 1 unit).
TICK: int = 1

# Number of shares per agent quote. Kept constant to simplify the action space.
QUOTE_QTY: int = 10

# Background liquidity: how many synthetic resting orders to seed each step.
BACKGROUND_ORDERS_PER_SIDE: int = 5

# Background order quantity (shares per synthetic order).
BACKGROUND_QTY: int = 20

# Price range (in ticks from mid) over which background orders are spread.
BACKGROUND_SPREAD_TICKS: int = 20

# Poisson arrival rate λ for synthetic market orders per step.
FLOW_LAMBDA: float = 1.2

# Size of each synthetic market order (shares).
FLOW_QTY: int = 5


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ──────────────────────────────────────────────────────────────────────────────

class MarketMakerEnv(gym.Env):
    """
    Single-instrument market-making environment backed by a C++ LOB engine.

    Action Space
    ────────────
    Discrete(4):
        0 — Hold / Cancel:  Cancel any open agent quotes. Do not re-quote.
        1 — Quote Tight:    Bid at (mid - 1 tick), Ask at (mid + 1 tick).
        2 — Quote Wide:     Bid at (mid - 3 ticks), Ask at (mid + 3 ticks).
        3 — Clear Inventory: Fire a market order to return inventory to zero.

    Observation Space
    ─────────────────
    Box(9, dtype=float32):
        [0]   Agent inventory  (signed shares; negative = short)
        [1]   Mid-price        (cents)
        [2]   Bid-ask spread   (cents; 0 if one side is empty)
        [3]   Bid level 1 total volume (best bid)
        [4]   Bid level 2 total volume
        [5]   Bid level 3 total volume
        [6]   Ask level 1 total volume (best ask)
        [7]   Ask level 2 total volume
        [8]   Ask level 3 total volume

    Reward
    ──────
        r_t = [W_t - W_{t-1}] - λ * inventory_t²

        where  W_t = cash_t + inventory_t × mid_t
    """

    metadata = {"render_modes": ["human"]}

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        instrument_id: str = "SIM",
        initial_mid: int = 10_000,       # $100.00 in cents
        episode_length: int = 500,       # steps per episode
        max_inventory: int = 200,        # absolute inventory cap (shares)
        inventory_penalty: float = 0.01, # λ in the reward function
        flow_lambda: float = FLOW_LAMBDA,
        tick_size: int = TICK,
        quote_qty: int = QUOTE_QTY,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # ── Configuration ─────────────────────────────────────────────────────
        self.instrument_id     = instrument_id
        self.initial_mid       = initial_mid
        self.episode_length    = episode_length
        self.max_inventory     = max_inventory
        self.inventory_penalty = inventory_penalty
        self.flow_lambda       = flow_lambda
        self.tick_size         = tick_size
        self.quote_qty         = quote_qty
        self.render_mode       = render_mode

        # ── Action Space: Discrete(4) ─────────────────────────────────────────
        self.action_space = spaces.Discrete(4)

        # ── Observation Space: Box(9) ─────────────────────────────────────────
        #
        #  Lower bounds: inventory can be negative; volumes and prices >= 0.
        #  Upper bounds: use np.inf for unconstrained quantities.
        #  The inventory dimension is bounded symmetrically by max_inventory.
        obs_low = np.array(
            [-max_inventory,  # [0] inventory
              0.0,            # [1] mid-price
              0.0,            # [2] spread
              0.0, 0.0, 0.0,  # [3-5] bid volumes
              0.0, 0.0, 0.0], # [6-8] ask volumes
            dtype=np.float32,
        )
        obs_high = np.array(
            [max_inventory,         # [0] inventory
             np.inf,                # [1] mid-price
             np.inf,                # [2] spread
             np.inf, np.inf, np.inf, # [3-5] bid volumes
             np.inf, np.inf, np.inf], # [6-8] ask volumes
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Internal state (initialised properly in reset()) ──────────────────
        self.book: lob.OrderBook | None = None

        # Signed share count. Positive = long, negative = short.
        self.inventory: int = 0

        # Realised cash from completed trades (cents).
        self.cash: float = 0.0

        # Order IDs of the agent's currently resting quotes (None if absent).
        self._bid_order_id: int | None = None
        self._ask_order_id: int | None = None

        # Mid-price tracker: needed to compute mark-to-market wealth each step.
        self._last_mid: float = float(initial_mid)

        # Total wealth at the *start* of the current step (for ΔW calculation).
        self._last_wealth: float = 0.0

        # Step counter within the current episode.
        self._step_count: int = 0

        # Running mid-price (drifts with each step via _simulate_mid_drift).
        self._current_mid: int = initial_mid

    # ──────────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state and return the first
        observation.

        A fresh lob_engine.OrderBook is created on every reset so there is no
        state leakage between episodes. Background liquidity is seeded
        immediately so the first observation has a non-trivial LOB state.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ── Fresh order book ──────────────────────────────────────────────────
        self.book = lob.OrderBook(self.instrument_id)

        # ── Reset agent state ─────────────────────────────────────────────────
        self.inventory       = 0
        self.cash            = 0.0
        self._bid_order_id   = None
        self._ask_order_id   = None
        self._current_mid    = self.initial_mid
        self._last_mid       = float(self.initial_mid)
        self._last_wealth    = 0.0   # W_0 = cash(0) + inv(0) × mid(0) = 0
        self._step_count     = 0

        # ── Seed background liquidity ─────────────────────────────────────────
        #
        #  Without resting orders from other participants, the book would be
        #  empty and our quotes would never get filled. We place synthetic
        #  limit orders at multiple price levels to simulate a realistic ladder.
        self._seed_background_orders()

        # Discard the fill events produced by background order placement — the
        # agent has not acted yet and has no positions to track.
        self.book.drain_trades()

        obs  = self._get_observation()
        info = self._get_info()
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step of the environment.

        Step sequence
        ─────────────
          1. Apply the agent's action (cancel/quote/clear).
          2. Simulate synthetic market flow (random market orders).
          3. Drain the trade buffer and update inventory + cash.
          4. Refresh background liquidity for the next step.
          5. Advance mid-price with a small random walk.
          6. Compute reward = ΔWealth - λ·inventory².
          7. Build and return the observation.

        Parameters
        ──────────
        action : int
            One of {0, 1, 2, 3} — see class docstring.

        Returns
        ───────
        observation : np.ndarray, shape (9,)
        reward      : float
        terminated  : bool — episode ended because inventory limit was breached
        truncated   : bool — episode ended because max steps was reached
        info        : dict — diagnostic data (not used by the agent)
        """
        assert self.book is not None, "Call reset() before step()."

        # ── 1. Apply agent action ─────────────────────────────────────────────
        self._apply_action(action)

        # ── 2. Simulate synthetic market flow ─────────────────────────────────
        #
        #  Draw the number of arriving market orders from a Poisson distribution.
        #  This models uninformed ("noise") traders who lift or hit resting quotes.
        n_arrivals = np.random.poisson(self.flow_lambda)
        for _ in range(n_arrivals):
            self._simulate_market_order()

        # ── 3. Drain fills, update position ───────────────────────────────────
        self._process_fills()

        # ── 4. Refresh background liquidity ───────────────────────────────────
        #
        #  Background orders placed in the previous step may have been consumed
        #  by flow. Re-seed to maintain a stable-looking ladder.
        self._seed_background_orders()

        # Discard any fills from background order matching (not the agent's).
        # Note: agent fills were already drained in _process_fills() above.
        self.book.drain_trades()

        # ── 5. Advance mid-price (random walk) ────────────────────────────────
        self._advance_mid()

        # ── 6. Compute reward ─────────────────────────────────────────────────
        reward = self._compute_reward()

        # ── 7. Termination conditions ─────────────────────────────────────────
        self._step_count += 1

        # Terminated: agent breached the hard inventory limit.
        terminated = abs(self.inventory) >= self.max_inventory

        # Truncated: episode reached its maximum length.
        truncated = self._step_count >= self.episode_length

        obs  = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # RENDER
    # ──────────────────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Print a human-readable snapshot of the current environment state."""
        if self.render_mode != "human":
            return

        mid    = self.book.mid_price() or self._current_mid
        spread = self.book.spread() or 0
        bb     = self.book.best_bid()
        ba     = self.book.best_ask()

        print(
            f"Step {self._step_count:>4d} | "
            f"Mid ${mid / 100:.2f} | "
            f"Spread {spread}¢ | "
            f"BBO ${bb / 100:.2f} / ${ba / 100:.2f} | "
            f"Inv {self.inventory:+d} | "
            f"Cash ${self.cash / 100:,.2f} | "
            f"Wealth ${self._total_wealth() / 100:,.2f}"
            if bb and ba else
            f"Step {self._step_count:>4d} | Mid ${mid / 100:.2f} | "
            f"Inv {self.inventory:+d} | Cash ${self.cash / 100:,.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — ACTION HANDLER
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """
        Translate the discrete action index into order book operations.

        Actions
        ───────
        0 — Hold / Cancel:
              Cancel both open agent quotes (if any). No new quotes placed.
              The agent sits out this step, carrying its current inventory.

        1 — Quote Tight:
              Cancel existing quotes, then place new two-sided quotes
              1 tick away from mid on each side.

        2 — Quote Wide:
              Cancel existing quotes, then place new two-sided quotes
              3 ticks away from mid on each side.

        3 — Clear Inventory:
              Cancel quotes, then submit a market order of size |inventory|
              in the direction that returns inventory to zero. No-op if flat.
        """
        # Always cancel open quotes before acting. This prevents stale quotes
        # from accumulating and ensures the agent has at most one bid and one
        # ask resting at any time.
        self._cancel_agent_quotes()

        mid = self._current_mid

        if action == 0:
            # Hold — quotes already cancelled above; nothing more to do.
            pass

        elif action == 1:
            # Quote Tight: mid ± 1 tick
            self._place_agent_quotes(
                bid_price=mid - 1 * self.tick_size,
                ask_price=mid + 1 * self.tick_size,
            )

        elif action == 2:
            # Quote Wide: mid ± 3 ticks
            self._place_agent_quotes(
                bid_price=mid - 3 * self.tick_size,
                ask_price=mid + 3 * self.tick_size,
            )

        elif action == 3:
            # Clear Inventory: submit a market order to flatten the position.
            if self.inventory != 0:
                clear_qty  = abs(self.inventory)
                clear_side = lob.Side.ASK if self.inventory > 0 else lob.Side.BID
                # A market order on the ASK side sells shares (reduces long).
                # A market order on the BID side buys shares (covers short).
                self.book.add_order(
                    lob.Order(
                        price    = 0,           # Price unused for MARKET orders.
                        quantity = clear_qty,
                        side     = clear_side,
                        type     = lob.OrderType.MARKET,
                    )
                )

    def _cancel_agent_quotes(self) -> None:
        """Cancel the agent's open bid and ask quotes, if they exist."""
        if self._bid_order_id is not None:
            self.book.cancel_order(self._bid_order_id)
            self._bid_order_id = None

        if self._ask_order_id is not None:
            self.book.cancel_order(self._ask_order_id)
            self._ask_order_id = None

    def _place_agent_quotes(self, bid_price: int, ask_price: int) -> None:
        """
        Place a two-sided resting quote pair and record their order IDs.

        Guard rails
        ───────────
        - Prices are clipped to be strictly positive (the engine uses 0 as the
          market-sell sentinel price, so resting orders at price=0 are unsafe).
        - The bid price is clipped to always be strictly less than the ask price
          to avoid self-crossing (which would generate an immediate fill and
          leave the agent with a wash trade).
        """
        bid_price = max(1, bid_price)
        ask_price = max(bid_price + self.tick_size, ask_price)

        self._bid_order_id = self.book.add_order(
            lob.Order(
                price    = bid_price,
                quantity = self.quote_qty,
                side     = lob.Side.BID,
                type     = lob.OrderType.LIMIT,
            )
        )
        self._ask_order_id = self.book.add_order(
            lob.Order(
                price    = ask_price,
                quantity = self.quote_qty,
                side     = lob.Side.ASK,
                type     = lob.OrderType.LIMIT,
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — MARKET SIMULATION
    # ──────────────────────────────────────────────────────────────────────────

    def _seed_background_orders(self) -> None:
        """
        Place synthetic resting limit orders on both sides of the book.

        These represent liquidity from other market participants and ensure the
        LOB always has a non-trivial depth ladder. Without them the book would
        often be empty and `mid_price()` would return None.

        Orders are placed at uniformly spaced price levels spanning
        [mid + 1, mid + BACKGROUND_SPREAD_TICKS] ticks for the ask side and
        [mid - BACKGROUND_SPREAD_TICKS, mid - 1] for the bid side.
        """
        mid = self._current_mid

        for i in range(1, BACKGROUND_ORDERS_PER_SIDE + 1):
            offset = i * (BACKGROUND_SPREAD_TICKS // BACKGROUND_ORDERS_PER_SIDE)

            # Background bid: below mid
            bid_px = max(1, mid - offset)
            self.book.add_order(
                lob.Order(
                    price    = bid_px,
                    quantity = BACKGROUND_QTY,
                    side     = lob.Side.BID,
                    type     = lob.OrderType.LIMIT,
                )
            )

            # Background ask: above mid
            ask_px = mid + offset
            self.book.add_order(
                lob.Order(
                    price    = ask_px,
                    quantity = BACKGROUND_QTY,
                    side     = lob.Side.ASK,
                    type     = lob.OrderType.LIMIT,
                )
            )

    def _simulate_market_order(self) -> None:
        """
        Generate one synthetic market order (buy or sell with equal probability).

        This is the primary mechanism by which the agent's resting quotes get
        filled: the synthetic flow order hits the best available quote on the
        side it is trading against, which may or may not be the agent's quote
        depending on its price relative to the background orders.

        The market order size is fixed at FLOW_QTY shares. Extending to a
        random size (e.g., log-normal) would produce more realistic fill
        patterns.
        """
        # 50/50 uninformed buy or sell.
        side = lob.Side.BID if random.random() < 0.5 else lob.Side.ASK
        self.book.add_order(
            lob.Order(
                price    = 0,
                quantity = FLOW_QTY,
                side     = side,
                type     = lob.OrderType.MARKET,
            )
        )

    def _advance_mid(self) -> None:
        """
        Apply a discrete-time random walk to the mid-price.

        Each step the mid moves ±1 tick with equal probability (and stays flat
        with probability 1/3). This is a simplified Brownian motion on the
        price grid. A more realistic model would use a calibrated drift or
        mean-reversion process.
        """
        move = random.choice([-self.tick_size, 0, self.tick_size])
        self._current_mid = max(1, self._current_mid + move)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — FILLS PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _process_fills(self) -> None:
        """
        Drain the lob_engine trade buffer and update the agent's position.

        For each fill:
          - If the agent's bid order was involved  → agent bought shares:
                inventory += exec_qty
                cash      -= exec_qty × exec_price
          - If the agent's ask order was involved  → agent sold shares:
                inventory -= exec_qty
                cash      += exec_qty × exec_price

        Trades that do not involve the agent's order IDs are ignored (they are
        fills between background and flow orders).

        After a complete fill the agent's order ID is cleared so we don't
        attempt a redundant cancel next step.
        """
        trades = self.book.drain_trades()

        for trade in trades:
            agent_bid_filled = (trade.bid_order_id == self._bid_order_id)
            agent_ask_filled = (trade.ask_order_id == self._ask_order_id)

            if agent_bid_filled:
                # Agent's resting bid was hit — agent bought at exec_price.
                self.inventory += trade.exec_quantity
                self.cash      -= trade.exec_quantity * trade.exec_price
                # If the order was fully filled, clear the tracked ID.
                # (Partial fills leave the order in the book with a reduced qty.)
                order = self.book.find_order(self._bid_order_id)
                if order is None:
                    # Order no longer in book — fully consumed.
                    self._bid_order_id = None

            if agent_ask_filled:
                # Agent's resting ask was hit — agent sold at exec_price.
                self.inventory -= trade.exec_quantity
                self.cash      += trade.exec_quantity * trade.exec_price
                order = self.book.find_order(self._ask_order_id)
                if order is None:
                    self._ask_order_id = None

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — REWARD COMPUTATION
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Compute the step reward using the Avellaneda-Stoikov formulation.

        reward = ΔWealth - λ × inventory²

        ΔWealth = W_t - W_{t-1}

        where W_t = cash_t + inventory_t × mid_t  (mark-to-market total wealth)

        The quadratic inventory penalty λ × inventory² penalises the agent for
        holding directional risk. As |inventory| grows, the penalty increases
        quadratically, making it increasingly costly to hold a large position.

        Note: ΔWealth alone would reward the agent for collecting the spread
        but also for directional bets. The inventory penalty ensures the agent
        is specifically rewarded for providing liquidity while staying flat.
        """
        current_wealth = self._total_wealth()
        delta_wealth   = current_wealth - self._last_wealth

        inventory_penalty = self.inventory_penalty * (self.inventory ** 2)

        reward = delta_wealth - inventory_penalty

        # Store current wealth as the baseline for the next step.
        self._last_wealth = current_wealth

        return float(reward)

    def _total_wealth(self) -> float:
        """
        Mark-to-market total wealth in cents.

        W = cash + inventory × mid_price

        Uses the current LOB mid-price if available, or falls back to the
        tracked _current_mid if the book is momentarily one-sided.
        """
        mid = self.book.mid_price()
        if mid is None:
            mid = float(self._current_mid)
        return self.cash + self.inventory * mid

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — OBSERVATION & INFO
    # ──────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the 9-element observation vector.

        Layout
        ──────
        [0]   inventory     — signed share count
        [1]   mid_price     — cents; falls back to _current_mid if book is empty
        [2]   spread        — cents; 0 if one side is empty
        [3]   bid_vol_1     — total volume at best bid price level
        [4]   bid_vol_2     — total volume at 2nd best bid price level
        [5]   bid_vol_3     — total volume at 3rd best bid price level
        [6]   ask_vol_1     — total volume at best ask price level
        [7]   ask_vol_2     — total volume at 2nd best ask price level
        [8]   ask_vol_3     — total volume at 3rd best ask price level

        get_top_volumes() always returns exactly LOB_DEPTH elements (zero-padded
        if the book is shallow), so the observation shape is always (9,).
        """
        mid    = self.book.mid_price() or float(self._current_mid)
        spread = self.book.spread()    or 0.0

        bid_vols = self.book.get_top_volumes(lob.Side.BID, LOB_DEPTH)
        ask_vols = self.book.get_top_volumes(lob.Side.ASK, LOB_DEPTH)

        obs = np.array(
            [float(self.inventory), float(mid), float(spread)]
            + [float(v) for v in bid_vols]
            + [float(v) for v in ask_vols],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        """
        Return auxiliary diagnostic information.

        This dict is returned by step() and reset() but is not part of the
        agent's observation. Useful for logging, debugging, and evaluation.
        """
        mid = self.book.mid_price() or float(self._current_mid)
        return {
            "step"          : self._step_count,
            "inventory"     : self.inventory,
            "cash"          : self.cash,
            "mid_price"     : mid,
            "total_wealth"  : self._total_wealth(),
            "spread"        : self.book.spread() or 0,
            "open_orders"   : self.book.total_order_count(),
            "bid_order_id"  : self._bid_order_id,
            "ask_order_id"  : self._ask_order_id,
        }


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Minimal smoke test: run one full episode with random actions and print a
    summary. This verifies the environment is importable and internally
    consistent without requiring a trained agent.

    Run with:
        python MarketMakerEnv.py
    """
    import traceback

    print("=" * 60)
    print("  MarketMakerEnv — smoke test")
    print("=" * 60)

    env = MarketMakerEnv(render_mode="human")

    try:
        obs, info = env.reset(seed=42)
        print(f"\nInitial observation (shape={obs.shape}):\n  {obs}\n")

        total_reward = 0.0
        n_steps      = 0

        for _ in range(env.episode_length):
            action              = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps      += 1
            env.render()

            if terminated or truncated:
                break

        print("\n" + "=" * 60)
        print(f"  Episode finished after {n_steps} steps")
        print(f"  Total reward  : {total_reward:,.4f}")
        print(f"  Final wealth  : ${info['total_wealth'] / 100:,.2f}")
        print(f"  Final inventory: {info['inventory']} shares")
        print(f"  Final cash    : ${info['cash'] / 100:,.2f}")
        print(f"  Terminated    : {terminated}")
        print(f"  Truncated     : {truncated}")
        print("=" * 60)

    except Exception:
        traceback.print_exc()
    finally:
        env.close()
