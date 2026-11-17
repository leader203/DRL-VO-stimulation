# Test fixtures

The `native_` prefix identifies observations captured in Isaac Sim with ROS 2
navigation. The tests replay these frozen inputs on CPU and record where the
local replay ends. Source paths in fixture provenance identify the historical
run.

## native_heading_tangent_20261105.npz

Static ROS raster and the three measured roots, headings, initial velocities and
13 stationary physical obstacle footprints at 44 simulation seconds from
`tests/output/avoidance_heading_20261105/native120_v2/runtime`. The corresponding
trace failed at 45.43333333333204 s after `walker_02` made no route progress for
12 s. A large soft wall force favored a sideways velocity; the forward ORCA
query retained a nonzero tangent velocity which decayed without approaching its
visible local guide `[-2.83, -0.49, 0]` [m].

The isolated regression makes only `walker_02` walk to that recorded guide and
holds the other two people at their measured roots. It checks three seconds of
actual motion, turn rate and swept static/dynamic/peer clearance. It does not
replay the original goal queues, door controllers or complete user routes.
`snapshot_json` records the source trace/map SHA256, measured SI data and active
navigation settings; the raster is int8 and loads with `allow_pickle=False`.
The fixture is portable and requires no source USD, SDK or runtime output.

`late_snapshot_json` adds the successful 48 s observation from
`tests/output/avoidance_heading_20261105/native120_v3/runtime`; it uses the same
byte-identical static raster. That native run failed at 49.1333333333 s
after another route-progress timeout. Reverse-only recovery no longer triggered
when the raw ORCA velocity had a positive but ineffective route component. This
second isolated case retains its measured roots, initial velocities and all
13 stationary obstacle footprints, then applies the same three-second progress
and exact swept-clearance assertions. It is not sampled from the failed command.

## native_passing_ready_20261105.npz

Static ROS raster and three-person/13-obstacle request at 19.033333333333534 s
from `tests/output/avoidance_heading_20261105/native120/runtime`. The yielder
was 0.01193 m short of its fixed refuge: the hypothetical endpoint permitted
a complete passing path, but the actual body still blocked the narrow route.
The portable JSON scalar retains the measured states, original goals/tolerance,
observed footprints and original transaction origin/deadline. No SDK, source USD
or runtime output is required; loading uses `allow_pickle=False`.
The same raster also contains the 17.15 s transaction-selection observations:
the original 24-direction search had a certified 240-degree refuge with a
4.8-degree body turn, but its side-only top-four filter selected a 75-degree
refuge requiring a 160.2-degree turn instead.

## native_goal_queue_20261104.npz

Exact static ROS occupancy raster extracted from
`tests/output/goal_door_native_20261104_v4/native/runtime/map.npz` for the
2026-11-04 native goal-queue regression. The first `walker_03` goal wait occurred
at simulation time 25.066666666666524 s. The associated test supplies the
original immutable route inputs; the fixture stores the raster and v5/v6 observations.

Fields: `occupancy` is int8, shape (313, 593); `resolution_m` is a scalar float64
with value 0.02 m; `origin_xy_m` is float64, shape (2,), with value [-6.86, -3.6] m.
The compressed NPZ contains no pickled objects and loads with `allow_pickle=False`.
Raw occupancy SHA256:
`06eec8a965d1fcb7a16c27ee382760de54fc54123454118c3eee358e7387deec`.

The v5 payload was extracted from
`tests/output/goal_door_native60_20261104_v5/runtime/trace.jsonl` and its bootstrap.
It adds all 13 measured obstacle footprints/velocities,
all three observed people, and the four automatic-door descriptors at the same
25.0666667 s event. This includes the stationary, uncontrolled `Door_0000`
that invalidated the former static-only detour. `observation_json` is UTF-8 JSON
stored as a NumPy Unicode scalar; every fixture load uses `allow_pickle=False`.
The map bytes are identical to v4. No scene USD or runtime output is required.

The `return_observation_json` payload fixes the v6 return-flow failure at
53.19999999999826 s, using its last failed request and previous public holder
record. It includes the original three plans/spawns, all 13 actual obstacles
and the actor/goal state. The regression checks local queue clearance and
continued original-route motion; it does not freeze remote door feedback for
a whole future episode. The original scene map is byte-identical.

## native_goal_departure_swap_20261105.json

The exact three-person/13-footprint failed request at 23.83333333333326 s from
`tests/output/turning_fix_20261105/native120/runtime`. The frozen solver SHA256
was `42655b113ef70456f496d45c27f6865443f1b5b05375563c929c50f594443c50` and
goal coordinator SHA256 was
`10878b1950ad63ea75718a541aace26afaf30a07c711f921d515438d238943d9`.
The arriving owner had just started its original turn and two-second dwell.
The approaching person's measured position was safe, but no sampled bounded
newcomer refuge certified the owner's complete exit. Once its original dwell
finished, a small owner refuge permitted both unchanged routes to continue.

The JSON includes immutable original plans, spawns, effective settings, measured
obstacles/door feedback, and route phases/deadlines reconstructed by replaying
the prior recorded observations. Its request SHA256 and raw raster SHA256 are
recorded under `provenance`. Occupancy, resolution and origin were checked equal
to `native_goal_queue_20261104.npz`; the test reuses that portable raster with
`allow_pickle=False`, without loading source USD or runtime output. Frozen future
door feedback limits this fixture to the local queue/owner recovery, rather than
an entire future simulation episode.

## native_passing_third_arrival_20261105.json

Controller state immediately after the first passing frame at
84.56666666666315 s from `tests/output/turning_fix_20261105/native120_v6`.
The old source later failed at 93.91666666666262 s: an unowned third route
entered the two-person passage, leaving no full path for the passer. Simply
holding everyone at the failed snapshot also timed out at the original
transaction deadline. Holding the still-safe third person before entry retained
the original certified route and allowed all three pending original goals.

The JSON stores the original plans/spawns/settings, measured three-person
states, all 13 obstacle footprints, measured door feedback, current route and
passing lifecycle state, and original source fingerprints. These states came
from an exact recorded-observation replay; its maximum command-position error
at the failed request was 2.5e-11 m. The raster was checked byte-equal to
`native_goal_queue_20261104.npz`, loaded with `allow_pickle=False`. The fixture
requires no runtime output, SDK, source USD or pickle. Frozen future door
feedback limits the regression to the existing passage and local original
pending-goal completion; the complete native episode is verified separately.

## native_near_peer_heading_20261106.json

Actual three-body poses, velocities and 13 obstacle footprints from
`turning_followup_20261106/native120_v3` at 106.6 s. The 20 KB fixture starts a
fresh public `NavigationEngine` without saved private route/coordinator state.
Each current visible local guidance point is one local test goal; original
nominal speed, arrival tolerance, body dimensions, clearance and 60 Hz remain.
This isolates an ordinary near-peer encounter and does not replay complete
user business routes. Door resource metadata is omitted while actual observed
door footprints remain. The map arrays were checked exactly equal to reused
`native_heading_tangent_20261105.npz`; loading requires no SDK, USD, output
directory or pickle. Provenance and map SHA256 are stored in the JSON.

The old reciprocal recovery query caused repeated heading reversals toward a
fixed visible point. The two-second public regression retains complete static,
dynamic and pair swept-chord certificates, movement/heading bounds, actual
arrival and unchanged plans, while limiting excess back-and-forth rotation.

## Other recorded navigation cases

| Fixture | CPU regression boundary |
|---|---|
| `native_already_departing_goal_20261106.json` | The goal owner has begun departing. A bounded recovery clears the queue while retaining the original plans and goals. |
| `native_opening_refuge_20261106.json` | First measured door refuge through first authorization to open. The closed door footprint remains fixed during the replay. |
| `native_parking_side_step_20261106.json` | Three-second near-wall encounter toward the recorded active targets. Future obstacle and door observations remain fixed. |
| `native_passing_external_blocker_20261106.json` | Three pending goals with an observed blocker. The replay keeps original waits and headings and stops at those goals. |

These four snapshots reuse the raster from `native_goal_queue_20261104.npz`.

## native_parked_follower_20261109.npz

Static ROS map arrays and the two initial actors involved in the parked-following
episode in `overhead120_20261109_1238/recording_verified`. The test uses the
public engine with their saved routes and 60 Hz timing, independently of Kit,
ROS, the output directory, and the other two actors. The map, plans, and initial
states also match `portal_queue_overhead120_20261109_hhwpxD/recording`. The test
uses the current 0.0275 m map and 0.225 m pair clearances and requires the first
actor to pass the stationary peer before its authored release, with exact static
and pair swept clearance throughout.

## native_cached_corner_20261109.npz

The static raster and the four observed actor states, live obstacles, and door
feedback at 56.85 s from `planning_footprint_20261109/native120_quarter_cm_verified`.
The fixture also stores original plans and cached route fields reconstructed by
replaying that run to the selected frame. Its JSON includes hashes of the map,
bootstrap, native result and original trace row. A fresh CPU engine runs the
next half-second with the observed door and obstacle geometry held fixed. The
test checks progress away from the open jamb and static, dynamic and peer swept
clearance without requiring the native output directory or a serialized engine.
