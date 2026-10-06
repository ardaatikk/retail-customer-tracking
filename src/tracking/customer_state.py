from dataclasses import dataclass
from math import hypot

from src.tracking.tracker import Track


@dataclass
class CustomerState:
    local_track_id: int

    first_seen: float
    last_seen: float

    start_ground_x: float
    start_ground_y: float

    max_displacement: float = 0.0
    movement_frames: int = 0

    previous_ground_x: float = 0.0
    previous_ground_y: float = 0.0

    confidence_sum: float = 0.0
    confidence_count: int = 0

    confirmed: bool = False
    customer_id: int | None = None
    
    current_zone: str | None = None
    zone_entered_at: float | None = None
    zone_dwell_times: dict[str, float] | None = None
    zone_history: list[str] | None = None

    @property
    def observed_time(self) -> float:
        return self.last_seen - self.first_seen
    
    def elapsed_time(
        self,
        current_timestamp: float,
    ) -> float:
        return max(
            0.0,
            current_timestamp - self.first_seen,
        )

    @property
    def average_confidence(self) -> float:
        if self.confidence_count == 0:
            return 0.0

        return (
            self.confidence_sum
            / self.confidence_count
        )
        
    def update_zone(
        self,
        zone: str | None,
        timestamp: float,
    ) -> None:
        if zone is None:
            return

        if self.zone_dwell_times is None:
            self.zone_dwell_times = {}

        if self.zone_history is None:
            self.zone_history = []

        # First known zone.
        if self.current_zone is None:
            self.current_zone = zone
            self.zone_entered_at = timestamp
            self.zone_history.append(zone)
            return

        # Still in the same zone.
        if zone == self.current_zone:
            return

        # Customer moved to another zone.
        if self.zone_entered_at is not None:
            duration = (
                timestamp
                - self.zone_entered_at
            )

            self.zone_dwell_times[
                self.current_zone
            ] = (
                self.zone_dwell_times.get(
                    self.current_zone,
                    0.0,
                )
                + duration
            )

        self.current_zone = zone
        self.zone_entered_at = timestamp

        if (
            not self.zone_history
            or self.zone_history[-1] != zone
        ):
            self.zone_history.append(zone)


    def get_zone_dwell_times(
        self,
        timestamp: float,
    ) -> dict[str, float]:
        dwell_times = dict(
            self.zone_dwell_times or {}
        )

        if (
            self.current_zone is not None
            and self.zone_entered_at is not None
        ):
            dwell_times[
                self.current_zone
            ] = (
                dwell_times.get(
                    self.current_zone,
                    0.0,
                )
                + (
                    timestamp
                    - self.zone_entered_at
                )
            )

        return dwell_times


class CustomerStateManager:
    def __init__(
        self,
        min_observation_seconds: float = 0.5,
        min_displacement_pixels: float = 10.0,
        min_movement_frames: int = 5,
        movement_step_pixels: float = 2.0,
        min_average_confidence: float = 0.50,
    ) -> None:
        self.min_observation_seconds = (
            min_observation_seconds
        )

        self.min_displacement_pixels = (
            min_displacement_pixels
        )

        self.min_movement_frames = (
            min_movement_frames
        )

        self.movement_step_pixels = (
            movement_step_pixels
        )

        self.min_average_confidence = (
            min_average_confidence
        )

        self.states: dict[
            int,
            CustomerState,
        ] = {}

        self.next_customer_id = 1

    def update(
        self,
        track: Track,
        timestamp: float,
    ) -> CustomerState:

        track_id = track.local_track_id

        is_new_state = (
            track_id not in self.states
        )

        if is_new_state:
            self.states[track_id] = CustomerState(
                local_track_id=track_id,
                first_seen=timestamp,
                last_seen=timestamp,
                start_ground_x=track.ground_x,
                start_ground_y=track.ground_y,
                previous_ground_x=track.ground_x,
                previous_ground_y=track.ground_y,
                confidence_sum=track.confidence,
                confidence_count=1,
            )

        state = self.states[track_id]

        if not is_new_state:
            state.confidence_sum += (
                track.confidence
            )

            state.confidence_count += 1

        state.last_seen = timestamp

        displacement = hypot(
            track.ground_x
            - state.start_ground_x,
            track.ground_y
            - state.start_ground_y,
        )

        state.max_displacement = max(
            state.max_displacement,
            displacement,
        )

        frame_movement = hypot(
            track.ground_x
            - state.previous_ground_x,
            track.ground_y
            - state.previous_ground_y,
        )

        if (
            frame_movement
            >= self.movement_step_pixels
        ):
            state.movement_frames += 1

        state.previous_ground_x = (
            track.ground_x
        )

        state.previous_ground_y = (
            track.ground_y
        )

        if not state.confirmed:
            enough_time = (
                state.observed_time
                >= self.min_observation_seconds
            )

            enough_displacement = (
                state.max_displacement
                >= self.min_displacement_pixels
            )

            enough_movement_frames = (
                state.movement_frames
                >= self.min_movement_frames
            )

            enough_confidence = (
                state.average_confidence
                >= self.min_average_confidence
            )

            if (
                enough_time
                and enough_displacement
                and enough_movement_frames
                and enough_confidence
            ):
                state.confirmed = True

                state.customer_id = (
                    self.next_customer_id
                )

                self.next_customer_id += 1

        return state