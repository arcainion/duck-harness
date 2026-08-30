from __future__ import annotations

import unittest
from itertools import repeat

import numpy as np

from inference.agent.policy_pathfinding import (
    PATHFINDING_API_VERSION,
    POLICY_PATHFINDING_GLOBALS,
    action_destination,
    approach_points,
    cardinal_neighbors,
    clearance_mask,
    component_boxes,
    component_centers,
    connected_components,
    distance_map,
    find_cells,
    grid_line,
    line_of_sight,
    next_path_action,
    path_cost,
    path_is_valid,
    path_suffix,
    path_to_actions,
    reachable_points,
    shortest_path,
    shortest_path_to_any,
    shortest_path_through,
    shortest_approach_path,
    value_mask,
    weighted_shortest_path,
    weighted_shortest_path_to_any,
)


class PolicyPathfindingTests(unittest.TestCase):
    def test_cardinal_neighbors_are_bounded_passable_and_deterministic(self) -> None:
        passable = np.ones((3, 3), dtype=bool)
        passable[1, 2] = False
        self.assertEqual(
            ((0, 1), (2, 1), (1, 0)),
            cardinal_neighbors(passable, (1, 1)),
        )
        self.assertEqual(((0, 1), (1, 0)), cardinal_neighbors(passable, (0, 0)))

    def test_shortest_path_routes_around_obstacles(self) -> None:
        passable = np.ones((5, 5), dtype=bool)
        passable[0:4, 2] = False
        path = shortest_path(passable, (1, 0), (1, 4))
        self.assertEqual((1, 0), path[0])
        self.assertEqual((1, 4), path[-1])
        self.assertEqual(11, len(path))
        self.assertEqual(
            (
                "RIGHT",
                "DOWN",
                "DOWN",
                "DOWN",
                "RIGHT",
                "RIGHT",
                "UP",
                "UP",
                "UP",
                "RIGHT",
            ),
            path_to_actions(path),
        )

    def test_shortest_path_allows_actor_and_target_cells_outside_mask(self) -> None:
        passable = np.zeros((1, 3), dtype=bool)
        passable[0, 1] = True
        self.assertEqual(
            ((0, 0), (0, 1), (0, 2)),
            shortest_path(passable, (0, 0), (0, 2)),
        )

    def test_shortest_path_to_any_uses_bfs_order_to_break_ties(self) -> None:
        passable = np.ones((3, 3), dtype=bool)
        self.assertEqual(
            ((1, 1), (0, 1)),
            shortest_path_to_any(passable, (1, 1), ((2, 1), (0, 1))),
        )

    def test_unreachable_empty_goals_and_same_cell_results(self) -> None:
        passable = np.zeros((3, 3), dtype=bool)
        self.assertEqual((), shortest_path(passable, (0, 0), (2, 2)))
        self.assertEqual((), shortest_path_to_any(passable, (0, 0), ()))
        self.assertEqual(((1, 1),), shortest_path(passable, (1, 1), (1, 1)))

    def test_expansion_limit_bounds_search(self) -> None:
        passable = np.ones((1, 4), dtype=bool)
        self.assertEqual((), shortest_path(passable, (0, 0), (0, 3), 2))
        self.assertEqual(
            ((0, 0), (0, 1), (0, 2), (0, 3)),
            shortest_path(passable, (0, 0), (0, 3), 3),
        )
        for value in (0, 4097, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_expansions"):
                    shortest_path(passable, (0, 0), (0, 3), value)

    def test_invalid_grids_and_points_fail_cleanly(self) -> None:
        cases = (
            (np.ones((0, 2), dtype=bool), (0, 0), "non-empty"),
            (np.ones((65, 64), dtype=bool), (0, 0), "at most 4096"),
            (np.ones((2, 2), dtype=bool), (2, 0), "outside grid"),
            (np.ones((2, 2), dtype=bool), (False, 0), "integers"),
        )
        for passable, point, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    cardinal_neighbors(passable, point)

    def test_path_action_conversion_rejects_jumps_and_filters_invalid_action(
        self,
    ) -> None:
        path = ((2, 2), (2, 3), (1, 3))
        self.assertEqual(("RIGHT", "UP"), path_to_actions(path))
        self.assertEqual("RIGHT", next_path_action(path, ("RIGHT", "SPACE")))
        self.assertIsNone(next_path_action(path, ("UP", "SPACE")))
        self.assertIsNone(next_path_action(((2, 2),), ("RIGHT",)))
        with self.assertRaisesRegex(ValueError, "non-cardinal"):
            path_to_actions(((0, 0), (1, 1)))

    def test_find_cells_supports_multiple_values_and_bounded_results(self) -> None:
        grid = np.array([[3, 1, 3], [2, 3, 1]], dtype=np.uint8)
        self.assertEqual(((0, 0), (0, 2), (1, 0), (1, 1)), find_cells(grid, (2, 3)))
        self.assertEqual(((0, 1),), find_cells(grid, 1, max_results=1))
        self.assertEqual((), find_cells(grid, ()))
        with self.assertRaisesRegex(ValueError, "max_results"):
            find_cells(grid, 1, max_results=0)
        with self.assertRaisesRegex(ValueError, "at most 256"):
            find_cells(grid, repeat(1, 257))

    def test_distance_map_handles_multiple_sources_obstacles_and_is_read_only(
        self,
    ) -> None:
        passable = np.ones((3, 4), dtype=bool)
        passable[1, 1:3] = False
        distances = distance_map(passable, ((0, 0), (2, 3)))
        np.testing.assert_array_equal(
            np.array([[0, 1, 2, 2], [1, -1, -1, 1], [2, 2, 1, 0]], dtype=np.int16),
            distances,
        )
        self.assertEqual(np.int16, distances.dtype)
        self.assertFalse(distances.flags.writeable)
        with self.assertRaises(ValueError):
            distances[0, 0] = 4

    def test_distance_map_allows_blocked_origins_and_honors_expansion_limit(
        self,
    ) -> None:
        passable = np.ones((1, 5), dtype=bool)
        passable[0, 0] = False
        distances = distance_map(passable, ((0, 0),), max_expansions=2)
        self.assertEqual([0, 1, 2, -1, -1], distances[0].tolist())
        empty = distance_map(passable, (), max_expansions=2)
        self.assertTrue(np.all(empty == -1))

    def test_reachable_points_supports_radius_and_numpy_point(self) -> None:
        passable = np.ones((3, 3), dtype=bool)
        passable[0, 0] = False
        self.assertEqual(
            ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
            reachable_points(passable, np.array([1, 1]), max_distance=1),
        )
        with self.assertRaisesRegex(ValueError, "max_distance"):
            reachable_points(passable, (1, 1), max_distance=-1)

    def test_connected_components_are_size_sorted_and_filterable(self) -> None:
        passable = np.array(
            [
                [1, 1, 0, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 0],
            ],
            dtype=bool,
        )
        components = connected_components(passable)
        self.assertEqual((3, 2, 2), tuple(len(component) for component in components))
        self.assertEqual(((2, 0), (2, 1), (2, 2)), components[0])
        self.assertEqual((components[0],), connected_components(passable, min_size=3))

    def test_shortest_path_through_joins_legs_without_duplicate_waypoints(self) -> None:
        passable = np.ones((3, 4), dtype=bool)
        path = shortest_path_through(passable, (0, 0), ((0, 2), (2, 2)))
        self.assertEqual(((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)), path)
        self.assertEqual(((0, 0),), shortest_path_through(passable, (0, 0), ()))

    def test_shortest_path_through_fails_atomically_and_caps_waypoints(self) -> None:
        passable = np.zeros((3, 3), dtype=bool)
        self.assertEqual((), shortest_path_through(passable, (0, 0), ((0, 1), (2, 2))))
        with self.assertRaisesRegex(ValueError, "at most 32"):
            shortest_path_through(
                np.ones((1, 33), dtype=bool),
                (0, 0),
                tuple((0, index) for index in range(33)),
            )

    def test_coordinate_iterables_are_bounded_before_full_consumption(self) -> None:
        passable = np.ones((2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "at most 4096"):
            distance_map(passable, repeat((0, 0), 4097))
        with self.assertRaisesRegex(ValueError, "at most 4096"):
            path_to_actions(repeat((0, 0), 4097))

    def test_weighted_path_avoids_expensive_short_route(self) -> None:
        passable = np.ones((3, 5), dtype=bool)
        costs = np.ones((3, 5), dtype=np.float64)
        costs[1, 1:4] = 10.0
        path = weighted_shortest_path(passable, costs, (1, 0), (1, 4))
        self.assertEqual(
            ("UP", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "DOWN"),
            path_to_actions(path),
        )
        self.assertEqual(6.0, path_cost(costs, path))

    def test_weighted_path_to_any_selects_lowest_cost_not_fewest_steps(self) -> None:
        passable = np.ones((3, 4), dtype=bool)
        costs = np.ones((3, 4), dtype=np.float64)
        costs[1, 1] = 20.0
        path = weighted_shortest_path_to_any(passable, costs, (1, 0), ((1, 1), (1, 3)))
        self.assertEqual((1, 3), path[-1])
        self.assertEqual(5.0, path_cost(costs, path))

    def test_weighted_path_validates_cost_grid_and_expansion_budget(self) -> None:
        passable = np.ones((2, 3), dtype=bool)
        costs = np.ones((2, 3), dtype=np.float64)
        self.assertEqual((), weighted_shortest_path(passable, costs, (0, 0), (0, 2), 1))
        invalid_costs = (
            np.ones((3, 2), dtype=np.float64),
            np.array([[1, -1, 1], [1, 1, 1]], dtype=np.float64),
            np.array([[1, np.nan, 1], [1, 1, 1]], dtype=np.float64),
            np.full((2, 3), "bad"),
        )
        for invalid in invalid_costs:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaisesRegex(ValueError, "costs"):
                    weighted_shortest_path(passable, invalid, (0, 0), (0, 2))

    def test_path_cost_rejects_out_of_bounds_and_non_cardinal_paths(self) -> None:
        costs = np.ones((2, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "outside grid"):
            path_cost(costs, ((0, 0), (0, 2)))
        with self.assertRaisesRegex(ValueError, "non-cardinal"):
            path_cost(costs, ((0, 0), (1, 1)))

    def test_clearance_mask_accounts_for_edges_obstacles_and_is_read_only(self) -> None:
        passable = np.ones((7, 7), dtype=bool)
        passable[3, 3] = False
        cleared = clearance_mask(passable, radius=1)
        self.assertFalse(bool(np.any(cleared[0, :])))
        self.assertFalse(bool(np.any(cleared[:, 0])))
        self.assertFalse(bool(np.any(cleared[2:5, 2:5])))
        self.assertTrue(bool(cleared[1, 1]))
        self.assertFalse(cleared.flags.writeable)
        np.testing.assert_array_equal(passable, clearance_mask(passable, radius=0))
        with self.assertRaisesRegex(ValueError, "radius"):
            clearance_mask(passable, radius=9)

    def test_grid_line_and_line_of_sight_are_bounded_and_endpoint_tolerant(
        self,
    ) -> None:
        self.assertEqual(
            ((1, 1), (1, 2), (2, 3), (2, 4)),
            grid_line((1, 1), (2, 4), shape=(4, 5)),
        )
        passable = np.ones((4, 5), dtype=bool)
        passable[1, 2] = False
        self.assertFalse(line_of_sight(passable, (1, 1), (2, 4)))
        passable[1, 1] = False
        passable[2, 4] = False
        passable[1, 2] = True
        self.assertTrue(line_of_sight(passable, (1, 1), (2, 4)))
        with self.assertRaisesRegex(ValueError, "outside grid"):
            grid_line((0, 0), (4, 0), shape=(4, 5))

    def test_action_destination_projects_cardinal_moves_and_edges(self) -> None:
        self.assertEqual((1, 2), action_destination((1, 1), "right", (3, 3)))
        self.assertIsNone(action_destination((0, 1), "UP", (3, 3)))
        with self.assertRaisesRegex(ValueError, "one of UP"):
            action_destination((1, 1), "SPACE", (3, 3))

    def test_value_mask_is_read_only_and_supports_inversion(self) -> None:
        grid = np.array([[1, 2, 3], [2, 1, 0]], dtype=np.uint8)
        selected = value_mask(grid, (1, 2))
        np.testing.assert_array_equal(
            np.array([[1, 1, 0], [1, 1, 0]], dtype=bool), selected
        )
        np.testing.assert_array_equal(
            np.logical_not(selected), value_mask(grid, (1, 2), True)
        )
        self.assertFalse(selected.flags.writeable)
        with self.assertRaises(ValueError):
            selected[0, 0] = False

    def test_component_boxes_and_centers_preserve_component_order(self) -> None:
        mask = np.array(
            [
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 1, 0],
            ],
            dtype=bool,
        )
        self.assertEqual(
            ((0, 0, 1, 1, 3), (0, 4, 1, 4, 2), (2, 2, 2, 3, 2)),
            component_boxes(mask),
        )
        self.assertEqual(((0, 0), (0, 4), (2, 2)), component_centers(mask))
        self.assertEqual(((0, 0),), component_centers(mask, min_size=3))

    def test_approach_points_are_exact_distance_passable_and_row_major(self) -> None:
        passable = np.ones((5, 5), dtype=bool)
        passable[1, 2] = False
        self.assertEqual(
            ((2, 1), (2, 3), (3, 2)),
            approach_points(passable, ((2, 2),), distance=1),
        )
        self.assertEqual(
            ((0, 2), (1, 1), (1, 3), (2, 0), (2, 4), (3, 1), (3, 3), (4, 2)),
            approach_points(passable, ((2, 2),), distance=2),
        )
        with self.assertRaisesRegex(ValueError, "distance"):
            approach_points(passable, ((2, 2),), distance=9)

    def test_shortest_approach_path_stops_before_blocked_target(self) -> None:
        passable = np.ones((5, 5), dtype=bool)
        passable[2, 2] = False
        path = shortest_approach_path(passable, (2, 0), ((2, 2),))
        self.assertEqual(((2, 0), (2, 1)), path)
        self.assertNotIn((2, 2), path)
        self.assertEqual(
            (), shortest_approach_path(np.zeros((3, 3), dtype=bool), (0, 0), ((1, 1),))
        )

    def test_path_validation_allows_blocked_endpoints_but_not_interior(self) -> None:
        passable = np.ones((1, 4), dtype=bool)
        passable[0, 0] = False
        passable[0, 3] = False
        path = ((0, 0), (0, 1), (0, 2), (0, 3))
        self.assertTrue(path_is_valid(passable, path))
        passable[0, 2] = False
        self.assertFalse(path_is_valid(passable, path))
        self.assertFalse(path_is_valid(passable, ((0, 0), (0, 2))))
        self.assertFalse(path_is_valid(passable, ((0, 0), (0, 4))))
        self.assertFalse(path_is_valid(passable, ()))

    def test_path_suffix_uses_latest_occurrence_and_handles_missing_position(
        self,
    ) -> None:
        path = ((0, 0), (0, 1), (0, 0), (1, 0))
        self.assertEqual(((0, 0), (1, 0)), path_suffix(path, (0, 0)))
        self.assertEqual((), path_suffix(path, (2, 2)))
        with self.assertRaisesRegex(ValueError, "at most 4096"):
            path_suffix(repeat((0, 0), 4097), (0, 0))

    def test_public_helper_registry_is_immutable_and_complete(self) -> None:
        expected = {
            "approach_points",
            "component_boxes",
            "component_centers",
            "path_is_valid",
            "path_suffix",
            "shortest_approach_path",
            "value_mask",
        }
        self.assertEqual(1, PATHFINDING_API_VERSION)
        self.assertTrue(expected.issubset(POLICY_PATHFINDING_GLOBALS))
        self.assertEqual(1, POLICY_PATHFINDING_GLOBALS["PATHFINDING_API_VERSION"])
        self.assertTrue(
            all(callable(POLICY_PATHFINDING_GLOBALS[name]) for name in expected)
        )
        with self.assertRaises(TypeError):
            POLICY_PATHFINDING_GLOBALS["unsafe"] = object()


if __name__ == "__main__":
    unittest.main()
