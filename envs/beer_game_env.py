import functools
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box

# ==============================================================================
# CONFIGURABLE LEAD TIMES (this revision)
# ------------------------------------------------------------------------------
# Previously the three physical delays were hardcoded. They are now driven by the
# config, and EVERY default reproduces the old constant exactly -- so an existing
# config that does not set these keys behaves identically to before, and all prior
# checkpoints/runs remain valid. The three delays are:
#
#   1. ORDER lead time      : how long an agent's ORDER takes to reach its supplier
#                             (information delay up the chain). Old: manufacturer=1,
#                             everyone else=2.  Config: order_lead_time / _mfr.
#   2. SHIPPING lead time   : how long fulfilled GOODS take to travel down to the
#                             customer (physical delay). Old: fixed 2, or U(1,9) if
#                             the jittery_lead_time flag was set. Config:
#                             ship_lead_time (fixed) or ship_lead_time_range [lo,hi].
#   3. PRODUCTION lead time  : how long the manufacturer takes to PRODUCE goods it
#                             releases into its own shipping pipeline. Old: fixed 2.
#                             Config: production_lead_time / _range.
#
# Each delay can be a FIXED int, or a RANDOM range [lo, hi] sampled i.i.d. per
# shipment from the integers lo..hi inclusive. Setting a range is how you build
# "non-classical" scenarios (stochastic transit, supplier disruption) for training
# and OOD testing. Set narrow ranges == old behaviour; widen them for stress tests.
#
# OBSERVATION SCOPE: the AGENT observes only four local scalars (inventory, backlog,
# total on-order, last realised demand) -- no phased in-transit pipeline. The full
# physical pipeline remains available to the centralized CRITIC via get_global_state()
# (CTDE), but NOT any future demand oracle.
# REVISION 1/2: get_global_state() no longer appends next_d / next demand. Agents
# still receive post-step diagnostics through infos, including supply_chain_cost.
# HARD CONSTRAINT: any realised delay must still fit the critic's pipeline horizon, so
# __init__ validates every configured max delay is <= MAX_DELAY (the number of slots
# get_global_state() walks); a longer delay would land in a slot the critic cannot see.
# ==============================================================================

MAX_DELAY = 15  # pipeline horizon exposed by get_global_state(); see constraint above


def _is_strict_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)

def _is_strict_num(v):
    # Explicitly block NumPy complex types (np.complex64, np.complex128)
    if isinstance(v, bool) or isinstance(v, complex) or np.iscomplexobj(v):
        return False
    return isinstance(v, (int, float, np.number))


def _parse_lead_spec(cfg, fixed_key, range_key, default_fixed):
    """Return a sampler() -> int for a lead time.

    Resolution order (first match wins):
      * range_key present and a 2-list [lo, hi] -> sample integers lo..hi inclusive
      * fixed_key present                       -> constant int
      * else                                    -> default_fixed (the OLD constant)

    The returned closure takes the env's np_random so sampling stays reproducible
    under the episode seed. Validation of bounds happens in __init__, not here.
    """
    rng_spec = cfg.get(range_key, None)
    if rng_spec is not None:
        if (not isinstance(rng_spec, (list, tuple)) or len(rng_spec) != 2
                or not all(_is_strict_int(x) for x in rng_spec)):
            raise ValueError(f"{range_key} must be a [lo, hi] pair of ints, got {rng_spec}")
        lo, hi = int(rng_spec[0]), int(rng_spec[1])
        if lo <= 0 or hi < lo:
            raise ValueError(f"{range_key} needs 0 < lo <= hi, got [{lo}, {hi}]")
        # integers(lo, hi+1) -> inclusive of hi
        return (lambda rng, lo=lo, hi=hi: int(rng.integers(lo, hi + 1))), hi
    fixed = cfg.get(fixed_key, default_fixed)
    if not _is_strict_int(fixed) or fixed <= 0:
        raise ValueError(f"{fixed_key} must be a strictly positive int, got {fixed}")
    return (lambda rng, v=int(fixed): v), int(fixed)


class TransitPipeline:
    """A delay line for goods or orders in transit. `pipeline` maps an ARRIVAL step -> quantity:
    add_shipment(t, q, lead) schedules q to land at t+lead; receive_shipment(t) pops (and removes)
    whatever arrives at step t. Used for both the order pipeline (an order travelling UP to the
    supplier) and the shipment pipeline (goods travelling DOWN to the customer)."""
    def __init__(self):
        self.pipeline = {}

    def add_shipment(self, current_step, quantity, lead_time):
        if not _is_strict_int(current_step) or current_step < 0:
            raise ValueError(f"current_step must be non-negative int, got {type(current_step)}")
        if not _is_strict_num(quantity) or not np.isfinite(quantity) or quantity < 0:
            raise ValueError(f"Quantity must be finite and non-negative, got {quantity}")
        if not float(quantity).is_integer():
            raise ValueError(f"Quantity must be a discrete whole number, got {quantity}")
        if not _is_strict_int(lead_time) or lead_time <= 0:
            raise ValueError(f"Lead time must be a strictly positive integer, got {lead_time}")

        if int(quantity) == 0:
            return

        arr = current_step + int(lead_time)
        self.pipeline[arr] = self.pipeline.get(arr, 0) + int(quantity)

    def receive_shipment(self, current_step):
        if not _is_strict_int(current_step) or current_step < 0:
            raise ValueError("current_step must be a non-negative integer")
        return self.pipeline.pop(current_step, 0)


class BeerGameParallelEnv(ParallelEnv):
    metadata = {"name": "beer_game_v0"}

    def __init__(self, config):
        self.possible_agents = ["retailer", "wholesaler", "distributor", "manufacturer"]
        self.agents = self.possible_agents[:]

        self._config = config.copy() if config else {}
        # REVISION 3: remove the legacy/unused `lookahead` key from the env-owned
        # config. The current observation is intentionally four local scalars and
        # never exposes phased pipeline slots, so keeping `lookahead` here only
        # suggested behavior that no longer exists. Existing YAML files that still
        # contain the key remain loadable because we silently discard it.
        self._config.pop("lookahead", None)
        self.horizon = self._config.get("horizon", 50)
        self.max_order = self._config.get("max_order", 100)
        self.h = self._config.get("holding_cost", 0.5)
        self.b = self._config.get("backorder_cost", 1.0)

        if not _is_strict_int(self.horizon) or self.horizon <= 0: raise ValueError("Horizon must be pos int")
        if not _is_strict_int(self.max_order) or self.max_order <= 0: raise ValueError("Max order must be pos int")

        if not _is_strict_num(self.h) or not _is_strict_num(self.b): raise ValueError("Costs must be numeric")
        if not np.isfinite(self.h) or not np.isfinite(self.b) or self.h < 0 or self.b < 0: raise ValueError("Costs must be finite positive")

        # CANONICAL-COST FLAG (default off -> behavior unchanged; old checkpoints/runs stay valid).
        # When True, the backorder penalty is charged ONLY at the retailer (customer-facing) stage:
        # the canonical Clark-Scarf (1960) serial cost, for which an echelon base-stock policy is
        # PROVABLY optimal and the cost is provably convex. Default False = the beer-game team cost
        # (service at every echelon: each stage pays for its own backlog). Holding cost is charged
        # at every stage in BOTH modes (standard for the serial model).
        self._penalty_at_retailer_only = self._config.get("penalty_at_retailer_only", False)
        if type(self._penalty_at_retailer_only) is not bool:
            raise ValueError("penalty_at_retailer_only must be a strict boolean")

        # --- LEAD-TIME CONFIGURATION --------------------------------------------
        # REVISION 6: behavior is unchanged here; the corrected test suite adds
        # explicit coverage for fixed-width random ranges, invalid ranges, and the
        # MAX_DELAY guard so these config paths are now benchmark-safe.
        # Each call returns (sampler(rng)->int, max_possible_delay). Defaults below
        # reproduce the original hardcoded constants exactly:
        #   order:        non-manufacturer = 2, manufacturer = 1
        #   shipping:     2  (old jittery flag = U(1,9); now express that as a range)
        #   production:   2
        cfg = self._config
        self._order_lead, ord_max = _parse_lead_spec(cfg, "order_lead_time", "order_lead_time_range", 2)
        self._order_lead_mfr, ord_mfr_max = _parse_lead_spec(cfg, "order_lead_time_mfr", "order_lead_time_mfr_range", 1)
        self._production_lead, prod_max = _parse_lead_spec(cfg, "production_lead_time", "production_lead_time_range", 2)

        # Backward-compat shim for the old boolean `jittery_lead_time`: if it is set
        # True and no explicit shipping spec is given, reproduce the legacy U(1,9).
        jitter = cfg.get("jittery_lead_time", False)
        if type(jitter) is not bool:
            raise ValueError("jittery_lead_time must be a strict boolean")
        if jitter and "ship_lead_time" not in cfg and "ship_lead_time_range" not in cfg:
            self._ship_lead = lambda rng: int(rng.integers(1, 10))   # legacy U(1,9)
            ship_max = 9
        else:
            self._ship_lead, ship_max = _parse_lead_spec(cfg, "ship_lead_time", "ship_lead_time_range", 2)

        # Validate every max delay against the pipeline horizon the obs/state expose.
        self._max_lead = max(ord_max, ord_mfr_max, prod_max, ship_max)
        if self._max_lead > MAX_DELAY:
            raise ValueError(
                f"A configured max lead time ({self._max_lead}) exceeds MAX_DELAY ({MAX_DELAY}); "
                f"shipments would land in pipeline slots get_global_state() cannot see. "
                f"Lower the lead time or raise MAX_DELAY (and retrain: it changes the critic's state dim)."
            )

        demand_type = self._config.get("demand_type", None)
        if demand_type is None:
            demand_type = "step"
        valid_demands = ["step", "zero", "black_swan", "extreme_chaos", "poisson"]
        if demand_type not in valid_demands:
            raise ValueError(f"Invalid demand_type: {demand_type}")
        self._config["demand_type"] = demand_type

        self.np_random = np.random.default_rng()
        self._action_spaces = {a: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for a in self.possible_agents}
        # MINIMALIST OBSERVATION (Oroojlooyjadid-style, fully partially-observable):
        # [inventory, backlog, total_on_order, last_realized_incoming]. No pipeline phasing.
        obs_dim = 4
        self._observation_spaces = {a: Box(low=-2000.0, high=2000.0, shape=(obs_dim,), dtype=np.float32) for a in self.possible_agents}

    @property
    def config(self):
        """Read-only property preventing runtime mutation."""
        return self._config.copy()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None: self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.inventory = {a: 12 for a in self.possible_agents}
        self.backlog = {a: 0 for a in self.possible_agents}

        self.order_pipelines = {a: TransitPipeline() for a in self.possible_agents}
        self.shipment_pipelines = {a: TransitPipeline() for a in self.possible_agents}

        for a in self.possible_agents:
            self.shipment_pipelines[a].pipeline = {1: 4, 2: 4}
            self.order_pipelines[a].pipeline = {1: 4} if a == "manufacturer" else {1: 4, 2: 4}

        self.unfulfilled_orders = {a: sum(self.shipment_pipelines[a].pipeline.values()) + sum(self.order_pipelines[a].pipeline.values()) for a in self.possible_agents}

        self.stochastic_demand_cache = {}
        if self._config.get("demand_type") not in ["step", "zero"]:
            self.stochastic_demand_cache[1] = self._roll_stochastic_demand(1)
            self.stochastic_demand_cache[2] = self._roll_stochastic_demand(2)

        self.current_incoming_order = {a: 0 for a in self.possible_agents}
        self._period_demand = {a: 0 for a in self.possible_agents}
        self._period_demand_met = {a: 0 for a in self.possible_agents}
        return {a: self._build_obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def get_global_state(self):
        """True unclipped physical global state for CTDE.

        REVISION 1/2: Removed the former one-step-ahead demand channel (`next_d`).
        That value was a future exogenous demand oracle for the centralized critic.
        The decentralized agents never observed it, but allowing the critic to train
        with it would make DRACO/baseline comparisons unfair unless every method had
        the same forecast oracle. This state now contains only current physical
        variables and scheduled pipeline contents.

        Walks MAX_DELAY pipeline slots -- which is why every configured lead time
        must be <= MAX_DELAY (validated in __init__)."""
        state = [float(self.current_step)]
        for a in self.possible_agents:
            state.extend([self.inventory[a], self.backlog[a], self.unfulfilled_orders[a]])
            for t in range(1, MAX_DELAY + 1):
                state.append(self.shipment_pipelines[a].pipeline.get(self.current_step + t, 0))
                state.append(self.order_pipelines[a].pipeline.get(self.current_step + t, 0))

        return np.array(state, dtype=np.float32)

    def _roll_stochastic_demand(self, step):
        d_type = self._config.get("demand_type")
        # REVISION 5: The two stress regimes below are intentionally custom OOD
        # scenarios, not canonical Beer Game benchmark demand decks. Behavior is
        # unchanged; the comment prevents over-claiming these as literature-standard
        # demand processes.
        if d_type == "black_swan": return self.np_random.poisson(8 if step < 25 else 20)
        if d_type == "extreme_chaos":
            base = 8 if step < 10 else 30 if step < 20 else 0 if step < 30 else self.np_random.integers(5, 25)
            return self.np_random.poisson(base) if base > 0 else 0
        return self.np_random.poisson(8)

    def _peek_incoming_demand(self, agent, target_step):
        """Privileged lookup used ONLY by simulation mechanics and training-only
        channels. NOT called by _build_obs, so future demand never reaches the obs."""
        if agent == "retailer":
            d_type = self._config.get("demand_type")
            if d_type == "step": return 4 if target_step < 5 else 8
            if d_type == "zero": return 0
            if target_step not in self.stochastic_demand_cache:
                raise RuntimeError(f"Demand cache miss for step {target_step}. Observer effect detected.")
            return self.stochastic_demand_cache[target_step]
        idx = self.possible_agents.index(agent)
        return self.order_pipelines[self.possible_agents[idx - 1]].pipeline.get(target_step, 0)

    def _build_obs(self, agent):
        # MINIMALIST PARTIAL OBSERVABILITY. Four local scalars only:
        #   [inventory, backlog, total on-order, last realised incoming demand/order]
        # - 4th slot is the LAST realised demand (already observed), NOT a future peek.
        # - No phased in-transit pipeline is exposed. Inventory position
        #   (inv - backlog + on_order) is still recoverable from slots 0-2, so the
        #   base-stock head is unaffected; the agent loses only the ARRIVAL TIMING of
        #   in-transit goods, which it must now infer -- this is what turns variable
        #   lead times into a genuine hidden-dynamics test. The centralized critic still
        #   sees the full pipeline via get_global_state() (CTDE).
        last_inc = self.current_incoming_order[agent]
        obs = [float(self.inventory[agent]), float(self.backlog[agent]),
               float(self.unfulfilled_orders[agent]), float(last_inc)]
        return np.clip(np.array(obs, dtype=np.float32), -2000.0, 2000.0)

    def _validate_actions(self, actions):
        if not isinstance(actions, dict):
            raise ValueError(f"Actions must be a dict, got {type(actions)}")
        if set(actions.keys()) != set(self.agents):
            raise ValueError(f"Action keys mismatch. Expected {self.agents}")

        for agent in self.agents:
            act = actions[agent]
            if isinstance(act, str) or isinstance(act, bool) or isinstance(act, complex):
                raise ValueError("Action must be a numeric array")
            try:
                act_raw = np.array(act)
            except Exception:
                raise ValueError(f"Action for {agent} is invalid.")
            if act_raw.dtype == bool or not np.issubdtype(act_raw.dtype, np.number) or np.iscomplexobj(act_raw):
                raise ValueError(f"Action for {agent} must be a pure numeric array. Got dtype {act_raw.dtype}")
            act_array = act_raw.astype(float)
            if act_array.shape != (1,):
                raise ValueError(f"Action for {agent} must be shape (1,), got {act_array.shape}")
            if not np.isfinite(act_array[0]):
                raise ValueError(f"Action for {agent} must be finite")

    def step(self, actions):
        """Advance one period (week). The four phases below execute in this fixed order each step:
          PHASE 1  RECEIVE  -- goods scheduled to arrive now land in inventory.
          PHASE 2  FULFIL   -- each stage sees its incoming demand (retailer: customer demand;
                               others: the downstream stage's arriving orders), ships what it can
                               (rest -> backlog) DOWN the chain; the manufacturer turns received
                               orders into production (-> its own shipment pipe after production lead).
          PHASE 3  ORDER    -- each agent's action S becomes order = clip(S-IP); it travels UP to the
                               supplier after the order (information) lead.
          PHASE 4  COST     -- charge h*inventory (+ b*backlog, every stage by default / retailer only
                               if penalty_at_retailer_only); reward = -total_system_cost (shared).
        Returns the PettingZoo parallel tuple (obs, rewards, terminations, truncations, infos)."""
        if not self.agents:
            raise RuntimeError("Environment stepped after done")

        self._validate_actions(actions)

        self.current_step += 1
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        total_system_cost = 0.0

        # --- PHASE 1: RECEIVE INCOMING GOODS ---
        for agent in self.possible_agents:
            received = self.shipment_pipelines[agent].receive_shipment(self.current_step)
            self.inventory[agent] += received
            self.unfulfilled_orders[agent] -= received

        # --- PHASE 2: DETERMINE & FULFILL DEMAND ---
        for i, agent in enumerate(self.possible_agents):
            if agent == "retailer":
                current_demand = self._peek_incoming_demand(agent, self.current_step)
            else:
                current_demand = self.order_pipelines[self.possible_agents[i - 1]].receive_shipment(self.current_step)

            self.current_incoming_order[agent] = current_demand

            if agent == "manufacturer":
                requests = self.order_pipelines[agent].receive_shipment(self.current_step)
                if requests > 0:
                    # PRODUCTION lead time: how long the manufacturer takes to produce
                    # what it just received as an order, before it enters its shipping pipe.
                    prod_lt = self._production_lead(self.np_random)
                    self.shipment_pipelines[agent].add_shipment(self.current_step, requests, lead_time=prod_lt)

            backlog_prev = self.backlog[agent]
            total_req = current_demand + backlog_prev
            fulfilled = min(self.inventory[agent], total_req)
            self.inventory[agent] -= fulfilled
            self.backlog[agent] = total_req - fulfilled

            self._period_demand[agent] = current_demand
            self._period_demand_met[agent] = max(0, min(current_demand, fulfilled - backlog_prev))

            if agent != "retailer" and fulfilled > 0:
                # SHIPPING lead time: physical transit of goods down to the customer.
                ship_lt = self._ship_lead(self.np_random)
                self.shipment_pipelines[self.possible_agents[i - 1]].add_shipment(self.current_step, fulfilled, ship_lt)

        # --- PHASE 3: PLACE ORDERS ---
        for agent in self.agents:
            raw_action = float(np.clip(np.array(actions[agent], dtype=float)[0], 0.0, 1.0))
            order = int(np.floor(raw_action * self.max_order + 0.5))

            if self.current_step < self.horizon:
                self.unfulfilled_orders[agent] += order
                if order > 0:
                    # ORDER lead time: information delay for the order to reach the supplier.
                    if agent == "manufacturer":
                        lead_time = self._order_lead_mfr(self.np_random)
                    else:
                        lead_time = self._order_lead(self.np_random)
                    self.order_pipelines[agent].add_shipment(self.current_step, order, lead_time=lead_time)

        # --- PRE-ROLL FUTURE DEMAND ---
        next_lookahead = self.current_step + 2
        if self._config.get("demand_type") not in ["step", "zero"]:
            if next_lookahead not in self.stochastic_demand_cache:
                self.stochastic_demand_cache[next_lookahead] = self._roll_stochastic_demand(next_lookahead)

        # --- PHASE 4: ACCOUNTING & REWARDS ---
        done = self.current_step >= self.horizon

        # REVISION 1: agents may inspect the realized supply-chain cost AFTER the
        # round through infos. This is post-decision feedback, not a future-demand
        # signal. Existing keys are kept for backwards compatibility.
        local_costs = {}
        for agent in self.agents:
            # canonical (Clark-Scarf) cost charges the backorder penalty at the retailer only;
            # the default beer-game cost charges it at every echelon. Holding cost: every echelon.
            backorder_term = self.b * self.backlog[agent]
            if self._penalty_at_retailer_only and agent != "retailer":
                backorder_term = 0.0
            cost = (self.h * self.inventory[agent]) + backorder_term
            local_costs[agent] = cost
            total_system_cost += cost

        for agent in self.agents:
            # "demand" = demand realized THIS step (already observed) -> encoder's
            # self-supervised forecasting target (not in the obs the agent acted on).
            infos[agent] = {"local_cost": local_costs[agent],
                            "supply_chain_cost": total_system_cost,
                            "training_targets": {             # <--- NESTED FOR CLEAN ISOLATION
                                "demand": self._period_demand[agent],
                                "demand_met": self._period_demand_met[agent]
                            }
                     }

        for agent in self.agents:
            rewards[agent] = -total_system_cost
            terminations[agent] = False
            truncations[agent] = done

        obs_dict = {a: self._build_obs(a) for a in self.agents}
        if done: self.agents = []

        return obs_dict, rewards, terminations, truncations, infos