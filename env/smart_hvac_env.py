import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ML Imports
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ======================================================
# Comfort (Human Preference) Model
# ======================================================
class ComfortSetpointModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=900,
            random_state=42,
            warm_start=True
        )
        self.trained = False

    def train(self, X, y):
        if not self.trained:
            X_scaled = self.scaler.fit_transform(X)
            self.mlp.fit(X_scaled, y)
            self.trained = True
        else:
            X_scaled = self.scaler.transform(X)
            self.mlp.fit(X_scaled, y)

    def predict(self, x):
        if not self.trained:
            return 24.0
        x = np.atleast_2d(x)
        x_scaled = self.scaler.transform(x)
        return float(self.mlp.predict(x_scaled)[0])


# ======================================================
# Smart HVAC Environment
# ======================================================
class SmartHVACEnv(gym.Env):
    def __init__(
        self,
        timestep_minutes: int = 3,
        episode_hours: int = 10000,
        R: float = 2.0,
        C: float = 2.0,
        ac_power_kw: float = -8.5,
        occ_heat_per_person_kw: float = 0.05,
        base_setpoint: float = 24.0,
        comfort_band: float = 2.0,
        max_occupancy: int = 250,
        forecast_horizon: int = 3,
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.turbo_mode = False  # Default to normal physics
        # --- Time & Physics ---
        self.dt_min = timestep_minutes
        self.dt = timestep_minutes / 60.0
        self.episode_length_steps = int((episode_hours * 60) / timestep_minutes)
        self.R = R
        self.C = C
        self.ac_power_kw = ac_power_kw
        self.occ_heat_per_person_kw = occ_heat_per_person_kw

        # --- State ---
        self.T_in = 24.0
        self.T_out = 30.0
        self.setpoint = base_setpoint
        self.ac_on = True
        self.occupancy = max_occupancy // 2
        self.max_occupancy = max_occupancy
        self.manual_override = False
        self.comfort_band = comfort_band
        self.season = "Summer" # Default
        self.season_profiles = {
            "Summer":  {"base": 35.0, "amp": 5.0}, # Hot! 30-40°C
            "Monsoon": {"base": 28.0, "amp": 3.0}, # Humid/Warm 25-31°C
            "Autumn":  {"base": 22.0, "amp": 4.0}, # Mild 18-26°C
            "Winter":  {"base": 10.0, "amp": 5.0}, # Cold! 5-15°C
        }

        # --- Human override: None = agent control, True = force ON, False = force OFF
        self.forced_ac_state = None

        # --- ML Model ---
        self.comfort_model = ComfortSetpointModel()
        self.data_buffer_X = []
        self.data_buffer_y = []
        self._pretrain_comfort_model()

        # --- Observation Space ---
        obs_len = 3 + 2 * forecast_horizon + 4
        self.observation_space = spaces.Box(
            low=-50, high=60, shape=(obs_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # --- Rewards ---
        self.w_energy = 4.0
        self.w_deviation = 2.0
        self.w_stable = 1.0
        self.w_violation = 5.0
        self.E_max = abs(self.ac_power_kw) * self.dt
        self.dev_max = 5.0

        # --- Counters ---
        self.step_counter = 0
        self.ac_on_steps = 0
        self.ac_off_steps = 0

        # --- Time Tracking (IMPORTANT) ---
        self.total_elapsed_minutes = 0.0
        self.total_ac_on_minutes = 0.0

    # --------------------------------------------------
    def _pretrain_comfort_model(self):
        X, y = [], []
        for _ in range(500):
            t_out = self.rng.uniform(20, 40)
            occ = self.rng.integers(0, self.max_occupancy)
            hour = self.rng.integers(8, 20)
            pref = 24.0
            if t_out > 30: pref -= 1.0
            if occ > 100: pref -= 1.0
            X.append([t_out, occ, hour, 0])
            y.append(pref)
        self.comfort_model.train(X, y)

    # --------------------------------------------------
    def _get_obs(self):
        t_out_f = self.T_out + self.rng.normal(0, 0.1, 3)
        occ_f = np.clip(
            self.occupancy + self.rng.integers(-1, 2, 3),
            0, self.max_occupancy
        )
        return np.concatenate([
            np.array([self.T_in, self.T_out, self.setpoint], dtype=np.float32),
            t_out_f.astype(np.float32),
            occ_f.astype(np.float32),
            np.array([
                self.setpoint,
                float(self.ac_on),
                self.ac_on_steps / 1.0,
                self.ac_off_steps / 10.0
            ], dtype=np.float32)
        ])

    # --------------------------------------------------
    def _step_thermal_model(self):
        # 1. Heat Sources (Positive adds heat)
        # Heat from People
        occ_heat = self.occupancy * self.occ_heat_per_person_kw
        
        # Heat flowing In/Out through walls (Physics: Newton's Law of Cooling)
        # If T_out > T_in: Result is Positive (Room heats up)
        # If T_out < T_in: Result is Negative (Room cools down)
        heat_transfer = (self.T_out - self.T_in) / self.R
        
        # 2. Cooling Source (Negative removes heat)
        hvac_cooling = self.ac_power_kw if self.ac_on else 0.0

        # 3. Calculate Temperature Change (dT)
        # Total Power / Thermal Mass
        total_power_kw = heat_transfer + occ_heat + hvac_cooling
        dT = (total_power_kw * self.dt) / self.C

        # 4. Apply Change
        T_next = self.T_in + dT + self.rng.normal(0, 0.01)
        
        # Energy used (for Reward calculation)
        energy = abs(hvac_cooling) * self.dt if self.ac_on else 0.0
        
        return T_next, energy

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.T_in = 24.0
        self.T_out = 30.0
        self.setpoint = 24.0
        self.ac_on = True
        self.occupancy = self.max_occupancy // 2
        self.ac_on_steps = 0
        self.ac_off_steps = 0
        self.forced_ac_state = None

        # RESET TIME
        self.total_elapsed_minutes = 0.0
        self.total_ac_on_minutes = 0.0

        self.data_buffer_X = []
        self.data_buffer_y = []
        return self._get_obs(), {}

    # --------------------------------------------------
    def step(self, action):
        # --------------------------------------------------
        # 1. TIME & FEATURES
        # --------------------------------------------------
        current_hour = (self.step_counter * self.dt_min) // 60 % 24
        features = [self.T_out, self.occupancy, current_hour, 0]

        # --------------------------------------------------
        # 2. HUMAN FEEDBACK / TRUE PREFERENCE
        # --------------------------------------------------
        if self.manual_override:
            true_pref = self.setpoint
            interaction_penalty = -2.0
        else:
            true_pref = 24.0
            if self.T_out > 30:
                true_pref -= 1.0
            if self.occupancy > 100:
                true_pref -= 1.0
            true_pref += self.rng.normal(0, 0.1)
            interaction_penalty = 0.0

        # --------------------------------------------------
        # 3. ONLINE COMFORT MODEL UPDATE
        # --------------------------------------------------
        self.data_buffer_X.append(features)
        self.data_buffer_y.append(true_pref)

        if len(self.data_buffer_X) >= 10:
            self.comfort_model.train(self.data_buffer_X, self.data_buffer_y)
            self.data_buffer_X, self.data_buffer_y = [], []

        if not self.manual_override:
            self.setpoint = np.clip(
                self.comfort_model.predict(features), 18.0, 28.0
            )

        # --------------------------------------------------
        # 4. AGENT ACTION (AC TOGGLE)
        # --------------------------------------------------
        # We process the action immediately to see if AC *starts* ON or OFF.
        # --------------------------------------------------
        # 4. AGENT ACTION + HUMAN OVERRIDE (PRIORITY LOGIC)
        # --------------------------------------------------

        # Agent proposes an action
        if action == 3:
            proposed_ac_on = not self.ac_on
        else:
            proposed_ac_on = self.ac_on
        if self.forced_ac_state is None:
            if proposed_ac_on and self.T_in < self.setpoint:
                proposed_ac_on = False # Block the AC start
        
        # 3. Apply Final State
        # Manual Override still takes absolute priority over everything
        if self.forced_ac_state is not None:
            self.ac_on = self.forced_ac_state
        else:
            self.ac_on = proposed_ac_on
            
        # 4.5 Human Intervention Flag
        human_intervention = self.forced_ac_state is not None
        # Human override has absolute priority
        if self.forced_ac_state is not None:
            self.ac_on = self.forced_ac_state
        else:
            self.ac_on = proposed_ac_on
        # --------------------------------------------------
        # 4.5 HUMAN INTERVENTION FLAG
        # --------------------------------------------------
        human_intervention = self.forced_ac_state is not None


        # --------------------------------------------------
        # 5. SEASONAL OUTSIDE TEMPERATURE
        # --------------------------------------------------
        period = max(1, (24 * 60) // self.dt_min)
        profile = self.season_profiles.get(self.season, self.season_profiles["Summer"])

        self.T_out = (
            profile["base"]
            + profile["amp"] * math.sin(2 * math.pi * self.step_counter / period)
            + self.rng.normal(0, 0.2)
        )

        # --------------------------------------------------
        # 6. PHYSICS
        # --------------------------------------------------
        # CRITICAL FIX: Capture the state used for physics BEFORE it potentially changes
        # --------------------------------------------------
        # 6. PHYSICS (TUNED FOR REALISM)
        # --------------------------------------------------
        ac_active_during_step = self.ac_on 

        # --- A. Calculate Heat Sources ---
        # Heat flowing in from outside
        heat_gain_env = 0.05 * (self.T_out - self.T_in)
        
        # Heat from people (0.02 per 5 people)
        heat_gain_people = 0.02 * (self.occupancy / 5.0)
        
        net_heat_load = heat_gain_env + heat_gain_people

        # --- B. Calculate Cooling ---
        if ac_active_during_step:
            # 1. Standard Physics (Weaker, more realistic)
            # CHANGED: Reduced from -1.5 to -0.6 so it struggles in high heat
            base_cooling = -0.6 
            
            # 2. Turbo Logic Check
            if self.turbo_mode and net_heat_load > abs(base_cooling):
                # TURBO: Overpower the heat by 0.5 degrees
                cooling_power = -(net_heat_load + 0.5)
                power_kw = 2.5 
            else:
                # NORMAL: Uses the weaker base cooling
                cooling_power = base_cooling
                power_kw = 1.5
        else:
            cooling_power = 0.0
            power_kw = 0.0

        # --- C. Update Temperature ---
        T_next = self.T_in + net_heat_load + cooling_power
        T_next += self.rng.normal(0, 0.05) 
        
        # --- D. Calculate Energy ---
        energy = (power_kw * self.dt_min) / 60.0

        # --------------------------------------------------
        # 7. AUTO CUTOFF (AFTER COOLING)
        # --------------------------------------------------
        # If the AC was ON and cooled the room below setpoint, turn it OFF for the *next* step.
        if T_next <= self.setpoint:
            self.ac_on = False

        self.T_in = T_next

        # --------------------------------------------------
        # 8. TIMERS & COUNTERS
        # --------------------------------------------------
        # CRITICAL FIX: Update counters based on what happened *during* the step (ac_active_during_step)
        # not the state prepared for the *next* step (self.ac_on)
        self.total_elapsed_minutes += self.dt_min
        
        if ac_active_during_step:
            self.total_ac_on_minutes += self.dt_min
            self.ac_on_steps += 1
            self.ac_off_steps = 0
        else:
            self.ac_off_steps += 1
            self.ac_on_steps = 0

        # --------------------------------------------------
        # 9. REWARD
        # --------------------------------------------------
        error = T_next - true_pref
        half = self.comfort_band / 2

        dev = max(0, abs(error) - half) / self.dev_max
        within = abs(error) <= half

        r = (
            -self.w_energy * (energy / self.E_max)
            -self.w_deviation * dev
            + self.w_stable * within
            + interaction_penalty
        )
        # Penalize human intervention (preference shaping)
        if human_intervention:
            r -= 2.0

        # Use ac_active_during_step here too for consistency
        if error > 0.5 and not ac_active_during_step:
            r -= self.w_violation

        r = float(np.clip(r, -20, 20))

        # --------------------------------------------------
        # 10. TERMINATION & INFO
        # --------------------------------------------------
        self.step_counter += 1
        done = self.step_counter >= self.episode_length_steps

        return self._get_obs(), r, done, False, {
            "ac_on": self.ac_on,  # Send the UPDATED state (Post-Cutoff),
            "energy_kwh": energy,
            "ideal_temp": true_pref,
            "total_time_min": self.total_elapsed_minutes,
            "total_ac_on_min": self.total_ac_on_minutes,
            
            # --- NEW ---
            "human_override": human_intervention,
            "forced_ac_state": self.forced_ac_state
        }


    def set_season(self, season_name):
        if season_name in self.season_profiles:
            self.season = season_name
    