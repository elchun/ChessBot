# Script to simulate and parse a board
# Loosely based on https://github.com/RussTedrake/manipulation/blob/master/segmentation_data.py
from IPython.display import clear_output
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os.path as osp
import random


from plotly.offline import iplot

from stockfish import Stockfish

from pydrake.all import (
    RandomGenerator,
    Simulator,
    FixedOffsetFrame,
    Adder,
    AddMultibodyPlantSceneGraph,
    Demultiplexer,
    DiagramBuilder,
    InverseDynamicsController,
    MakeMultibodyStateToWsgStateSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PassThrough,
    RigidTransform,
    RollPitchYaw,
    SchunkWsgPositionController,
    StateInterpolatorWithDiscreteDerivative,
    Role,
    MeshcatVisualizer,
    RotationMatrix)

from chess_bot.utils.utils import colorize_labels

from chess_bot.resources.board import Board
from chess_bot.utils.path_util import get_chessbot_src
from chess_bot.utils.meshcat_utils import RobotMeshcatPoseSliders, RobotWsgButtonPanda, RobotStatus

from chess_bot.perception.icp_system import ICPSystem

from chess_bot.utils.plotly_utils import multiplot
from chess_bot.utils.game_station_utils import AddRgbdSensors, AddPanda, AddWsgPanda, AddPandaDifferentialIK


rng = np.random.default_rng()  # this is for python
generator = RandomGenerator(rng.integers(1000))  # for c++

class GameStation():
    def __init__(self, meshcat):
        """
        Class that simulates a chess board to play chess against a robot.
        """
        try:
            self.stockfish = Stockfish('/opt/homebrew/bin/stockfish')
        except:
            raise ValueError('Enter your own stockfish path')
        self.meshcat = meshcat
        builder = DiagramBuilder()

        self.station = builder.AddSystem(self.make_station())

        self.station_context = self.station.CreateDefaultContext()

        self.controller_plant = self.station.GetSubsystemByName(
            "panda_controller").get_multibody_plant_for_control()

        viz = MeshcatVisualizer.AddToBuilder(
            builder,
            self.station.GetOutputPort("query_object"),
            meshcat)

        meshcat.ResetRenderMode()
        meshcat.DeleteAddedControls()

        # dif_ik_frame = self.controller_plant.AddFrame(FixedOffsetFrame('panda_gripper_frame',
        #     self.controller_plant.GetFrameByName("panda_link8"),
        #     RigidTransform([0, 0, -0.2])))

        # Set up differential inverse kinematics.
        # differential_ik = AddPandaDifferentialIK(
        #     builder,
        #     self.controller_plant,
        #     frame=self.controller_plant.GetFrameByName("panda_link8"))

        differential_ik = AddPandaDifferentialIK(
            builder,
            self.controller_plant,
            frame=self.controller_plant.GetFrameByName('panda_gripper_frame'))
        builder.Connect(differential_ik.get_output_port(),
                        self.station.GetInputPort("panda_position"))
        builder.Connect(self.station.GetOutputPort("panda_state_estimated"),
                        differential_ik.GetInputPort("robot_state"))

        init_robot_value = RobotMeshcatPoseSliders.Value(roll=0.0,  # idk if this part works...
                                            pitch=0.0,
                                            yaw=0.0,
                                            x=0.0,
                                            y=0.0,
                                            z=0.5)
        self.robot_status = RobotStatus()

        # Set up teleop widgets.
        q0 = [0.0, 0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4]
        teleop = builder.AddSystem(
            RobotMeshcatPoseSliders(
                meshcat,
                self.robot_status,
                min_range=RobotMeshcatPoseSliders.MinRange(roll=0,
                                                        pitch=-0.5,
                                                        yaw=-np.pi,
                                                        x=-0.4,
                                                        y=-0.25,
                                                        z=0.0),
                max_range=RobotMeshcatPoseSliders.MaxRange(roll=2 * np.pi,
                                                        pitch=np.pi,
                                                        yaw=np.pi,
                                                        x=0.4,
                                                        y=0.25,
                                                        z=0.75),
                body_index=self.plant.GetBodyByName("panda_link8").index(),
                # It seems that value is set by the default joint positions, not here.
                value=init_robot_value))
        builder.Connect(teleop.get_output_port(0),
                        differential_ik.get_input_port(0))
        builder.Connect(self.station.GetOutputPort("body_poses"),
                        teleop.GetInputPort("body_poses"))

        wsg_teleop = builder.AddSystem(RobotWsgButtonPanda(meshcat, self.robot_status))

        builder.Connect(wsg_teleop.get_output_port(0),
                        self.station.GetInputPort("Schunk_Gripper_position"))
        builder.Connect(self.station.GetOutputPort("Schunk_Gripper_state_measured"),
                        wsg_teleop.GetInputPort('wsg_state_measured'))

        diagram = builder.Build()
        # context = diagram.CreateDefaultContext()

        # self.plant_context = self.plant.CreateDefaultContext()

        # plant.SetPositions(context_plant, plant.GetModelInstanceByName("panda"), q0)

        self.simulator = Simulator(diagram)
        context = self.simulator.get_mutable_context()

        self.mut_station_context = diagram.GetMutableSubsystemContext(self.station, context)

        self.mut_plant_context = diagram.GetMutableSubsystemContext(self.plant, context)

        # self.simulator.set_target_realtime_rate(1.0)
        self.simulator.set_target_realtime_rate(3.0)
        # self.simulator.set_publish_every_time_step(True)

        self.grasp_yaw = None

        meshcat.AddButton("Stop Simulation", "Escape")
        print("Press Escape to stop the simulation")

        self.simulator.AdvanceTo(0.01)

    def make_station(self, time_step=0.002):
        """
        Helper function to create the diagram for a chess board

        Args:
            time_step (float, optional): Timestep of plant. Defaults to 0.002.

        Returns:
            scene diagram?: Result of builder.Build().
        """
        builder = DiagramBuilder()

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                    time_step=time_step)

        self.inspector = self.scene_graph.model_inspector()
        self.parser = Parser(self.plant)

        self.board, board_idx, self.idx_to_location, self.instance_id_to_class_name = self._add_board()

        self.plant.set_stiction_tolerance(0.001)

        # Optimized for 1080 x 1920
        # X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.61, 0, -np.pi/2), [-0.65, 0, 0.44])
        # X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.75, 0, -np.pi/2), [-0.65, 0.2667, 0.6])  # for testing position
        X_Camera = RigidTransform(RollPitchYaw(-np.pi/2 + -0.77, 0, -np.pi/2), [-0.63, 0, 0.6])
        # X_Camera = RigidTransform(RollPitchYaw(np.pi/6, 0, -np.pi/2), [-0.6, 0, 0.4])
        camerabox_fn = osp.join(get_chessbot_src(), 'resources/models/camera_box.sdf')
        camera_instance = self.parser.AddModelFromFile(camerabox_fn, 'camera')
        camera_frame = self.plant.GetFrameByName('base', camera_instance)
        self.plant.WeldFrames(self.plant.world_frame(), camera_frame, X_Camera)
        # AddMultibodyTriad(camera_frame, scene_graph, length=0.1, radius=0.005)

        # Add robot system
        panda_idx = AddPanda(self.plant)
        wsg_idx = AddWsgPanda(self.plant, ModelInstanceIndex(panda_idx))


        self.plant.Finalize()

        self._set_default_board()
        self._connect_robot(builder, time_step, panda_prefix='panda', wsg_prefix='Schunk_Gripper')

        # Cameras and perception system
        camera_prefix = 'camera'
        port_dict = AddRgbdSensors(builder,
                    self.plant,
                    self.scene_graph,
                    model_instance_prefix=camera_prefix)

        pcd_port_0 = port_dict['pcd_ports'][0]

        # Add ICP System to first pcd port
        icp = builder.AddSystem(ICPSystem())
        builder.Connect(pcd_port_0.GetOutputPort('pcd_stack'),
            icp.GetInputPort('raw_pcd_stack'))

        builder.ExportOutput(icp.GetOutputPort('icp_pcd_stack'),
            'icp_pcd_stack')

        # Export "cheat" ports
        builder.ExportOutput(self.scene_graph.get_query_output_port(), "query_object")
        builder.ExportOutput(self.plant.get_contact_results_output_port(),
                            "contact_results")
        builder.ExportOutput(self.plant.get_state_output_port(),
                            "plant_continuous_state")
        builder.ExportOutput(self.plant.get_body_poses_output_port(), "body_poses")

        diagram = builder.Build()
        diagram.set_name("ManipulationStation")
        return diagram

    def _add_board(self):
        """
        Helper functoin to create and add a board to the station.

        Returns:
            board, board_idx, idx_to_location, instance_id_to_class_name: Extra
                information generated when the board is created.
        """
        board = Board()

        parser = Parser(self.plant)
        board_idx = parser.AddModelFromFile(osp.join(board.model_dir, board.board_fn))

        idx_to_location = {}

        instance_id_to_class_name = {}

        for location, piece in board.starting_board.items():
            name = location + piece  # Very arbitrary, may change later
            idx = parser.AddModelFromFile(osp.join(board.model_dir, board.piece_to_fn[piece]), name)

            # -- Add labels for label generation -- #
            frame_id = self.plant.GetBodyFrameIdOrThrow(
                self.plant.GetBodyIndices(idx)[0])
            geometry_ids = self.inspector.GetGeometries(frame_id, Role.kPerception)

            for geom_id in geometry_ids:
                instance_id_to_class_name[int(
                    self.inspector.GetPerceptionProperties(geom_id).GetProperty(
                        "label", "id"))] = name

            idx_to_location[idx] = location

        # Weld the table to the world so that it's fixed during the simulation.
        board_frame = self.plant.GetFrameByName("board_body")
        # plant.WeldFrames(plant.world_frame(), board_frame, RigidTransform(RollPitchYaw(np.array([0, 0, np.pi/2])), [0, 0, 0]))
        self.plant.WeldFrames(self.plant.world_frame(), board_frame)

        # print(instance_id_to_class_name)
        return board, board_idx, idx_to_location, instance_id_to_class_name

    def _set_default_board(self):
        """
        Move the pieces to the default positions (start of chess game).
        """
        board_piece_offset = 0.0
        self.plant_context = self.plant.CreateDefaultContext()

        board_frame = self.plant.GetFrameByName("board_body")
        X_WorldBoard= board_frame.CalcPoseInWorld(self.plant_context)


        for idx, location in self.idx_to_location.items():
            piece = self.plant.GetBodyByName("piece_body", idx)
            x, y = self.board.get_xy_location(location)

            # Flip white pieces to face the right way
            if y < 0:
                yaw = np.pi
            else:
                yaw = 0

            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, yaw])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetDefaultFreeBodyPose(piece, X_BoardPiece)

    def _connect_robot(self, builder, time_step, panda_prefix='panda',
        wsg_prefix='Schunk_Gripper'):

        for i in range(self.plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            model_instance_name = self.plant.GetModelInstanceName(model_instance)
            # print('name: ', model_instance_name)
            if model_instance_name.startswith(panda_prefix):
                num_panda_positions = self.plant.num_positions(model_instance)

                # I need a PassThrough system so that I can export the input port.
                panda_position = builder.AddSystem(PassThrough(num_panda_positions))
                builder.ExportInput(panda_position.get_input_port(),
                                    model_instance_name + "_position")
                builder.ExportOutput(panda_position.get_output_port(),
                                    model_instance_name + "_position_commanded")
                # Export the iiwa "state" outputs.
                demux = builder.AddSystem(
                    Demultiplexer(2 * num_panda_positions, num_panda_positions))
                builder.Connect(self.plant.get_state_output_port(model_instance),
                                demux.get_input_port())
                builder.ExportOutput(demux.get_output_port(0),
                                    model_instance_name + "_position_measured")
                builder.ExportOutput(demux.get_output_port(1),
                                    model_instance_name + "_velocity_estimated")
                builder.ExportOutput(self.plant.get_state_output_port(model_instance),
                                    model_instance_name + "_state_estimated")

                # Make the plant for the iiwa controller to use.
                controller_plant = MultibodyPlant(time_step=time_step)
                controller_panda = AddPanda(controller_plant, collide=False)
                # Frame for placing control at gripper fingers
                # dif_ik_frame = controller_plant.AddFrame(FixedOffsetFrame('panda_gripper_frame',
                #     controller_plant.GetFrameByName("panda_link8"),
                #     RigidTransform([0.0, 0.02, 0.2])))

                dif_ik_frame = controller_plant.AddFrame(FixedOffsetFrame('panda_gripper_frame',
                    controller_plant.GetFrameByName("panda_link8"),
                    RigidTransform([0.0, 0.00, 0.2])))

                controller_plant.Finalize()

                kp = [600, 3000, 1000, 3000, 1000, 3000, 400]
                ki = [50, 50, 50, 50, 50, 50, 15]
                kd = [100, 300, 30, 300, 100, 300, 600]


                # Inverse dynamics
                panda_controller = builder.AddSystem(
                    InverseDynamicsController(controller_plant,
                                            kp=kp,
                                            ki=ki,
                                            kd=kd,
                                            has_reference_acceleration=False))
                panda_controller.set_name(model_instance_name + "_controller")
                builder.Connect(self.plant.get_state_output_port(model_instance),
                                panda_controller.get_input_port_estimated_state())


                # Add in the feed-forward torque
                adder = builder.AddSystem(Adder(2, num_panda_positions))
                builder.Connect(panda_controller.get_output_port_control(),
                                adder.get_input_port(0))
                # Use a PassThrough to make the port optional (it will provide zero
                # values if not connected).
                torque_passthrough = builder.AddSystem(
                    PassThrough([0] * num_panda_positions))
                builder.Connect(torque_passthrough.get_output_port(),
                                adder.get_input_port(1))
                builder.ExportInput(torque_passthrough.get_input_port(),
                                    model_instance_name + "_feedforward_torque")
                builder.Connect(adder.get_output_port(),
                                self.plant.get_actuation_input_port(model_instance))

                # Add discrete derivative to command velocities.
                desired_state_from_position = builder.AddSystem(
                    StateInterpolatorWithDiscreteDerivative(
                        num_panda_positions,
                        time_step,
                        suppress_initial_transient=True))
                desired_state_from_position.set_name(
                    model_instance_name + "_desired_state_from_position")
                builder.Connect(desired_state_from_position.get_output_port(),
                                panda_controller.get_input_port_desired_state())
                builder.Connect(panda_position.get_output_port(),
                                desired_state_from_position.get_input_port())

                # Export commanded torques.
                builder.ExportOutput(adder.get_output_port(),
                                    model_instance_name + "_torque_commanded")
                builder.ExportOutput(adder.get_output_port(),
                                    model_instance_name + "_torque_measured")

                builder.ExportOutput(
                    self.plant.get_generalized_contact_forces_output_port(
                        model_instance), model_instance_name + "_torque_external")

            elif model_instance_name.startswith(wsg_prefix):

                # Wsg controller.
                wsg_controller = builder.AddSystem(SchunkWsgPositionController())
                wsg_controller.set_name(model_instance_name + "_controller")
                builder.Connect(wsg_controller.get_generalized_force_output_port(),
                                self.plant.get_actuation_input_port(model_instance))
                builder.Connect(self.plant.get_state_output_port(model_instance),
                                wsg_controller.get_state_input_port())
                builder.ExportInput(
                    wsg_controller.get_desired_position_input_port(),
                    model_instance_name + "_position")
                builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                                    model_instance_name + "_force_limit")
                wsg_mbp_state_to_wsg_state = builder.AddSystem(
                    MakeMultibodyStateToWsgStateSystem())

                # wsg_mbp_force_to_wsg_force = builder.AddSystem(
                #     MakeMultibodyForceToWsgForceSystem()
                # )

                builder.Connect(self.plant.get_state_output_port(model_instance),
                                wsg_mbp_state_to_wsg_state.get_input_port())
                # builder.Connect(plant.get_force_output_port(model_instance),
                #                 wsg_mbp_force_to_wsg_force.get_input_port())
                builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                                    model_instance_name + "_state_measured")
                # builder.ExportOutput(wsg_mbp_force_to_wsg_force.get_output_port(),
                #                      model_instance_name + "_force_measured")

    def set_arbitrary_board(self, num_pieces=10):
        """
        Move <num_pieces> pieces to random locations on the board.  Make sure
        that no pieces are on top of eachother.

        Args:
            num_pieces (int, optional): number of pieces to show. Defaults to 10.
        """
        board_piece_offset = 0.0
        # plant_context = self.plant.GetMyMutableContextFromRoot(self.station_context)
        plant_context = self.mut_plant_context

        board_frame = self.plant.GetFrameByName("board_body")
        X_WorldBoard= board_frame.CalcPoseInWorld(plant_context)

        locations = set()  # set of (x, y) tuples

        indices = list(self.idx_to_location.keys())
        random.shuffle(indices)

        for idx in indices[:num_pieces]:
            piece = self.plant.GetBodyByName("piece_body", idx)
            # benchy = plant.GetBodyByName("benchy_body", name)

            # Find a position that is not currently occupied
            while True:
                x = np.random.randint(0, 8)
                y = np.random.randint(0, 8)
                if (x, y) not in locations:
                    break
            locations.add((x, y))
            # print('x: ', x, 'y: ', y)
            x, y = self.board.get_xy_location_from_idx(x, y)
            yaw = np.random.random() * 2 * np.pi
            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, yaw])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetFreeBodyPose(plant_context, piece, X_BoardPiece)

        # Put pieces off screen
        for i, idx in enumerate(indices[num_pieces:]):
            piece = self.plant.GetBodyByName("piece_body", idx)
            x, y = 1, 1 + i
            X_BoardPiece = RigidTransform(
                RollPitchYaw(np.asarray([0, 0, 0])), p=[x, y, board_piece_offset])
            X_BoardPiece = X_WorldBoard.multiply(X_BoardPiece)
            self.plant.SetFreeBodyPose(plant_context, piece, X_BoardPiece)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1.0)

    # -- Functions for gameplay -- #
    def play_game(self, pre_move_list: list[str]=[]):
        # castle using the king coordinates

        prev_board = deepcopy(Board.starting_board_list)
        self.robot_status.set_pose_value(z=0.4)

        if pre_move_list:
            print('Setting from initial state!')
        for move in pre_move_list:
            start_pos_str = move[:2]
            end_pos_str = move[2:]
            start_pos = Board.location_to_coord(start_pos_str)
            end_pos = Board.location_to_coord(end_pos_str)

            # Check if we're castling
            moves_made = self.move_castle(start_pos, end_pos, prev_board, use_robot=False)

            if not moves_made:
                self.make_move(start_pos, end_pos)
                moves_made = [(start_pos, end_pos)]


            for _start_pos, _end_pos in moves_made:
                prev_board[_end_pos[0]][_end_pos[1]] = prev_board[_start_pos[0]][_start_pos[1]]
                prev_board[_start_pos[0]][_start_pos[1]] = '  '

            self.stockfish.make_moves_from_current_position([move])

        if pre_move_list:
            print('Done setting board!')


        while True:
            move_correct = False
            while not move_correct:
                # Get player move (Player is always white)
                try:
                    player_start_pos, player_end_pos = self._get_player_move()
                except ValueError:
                    print('Exiting!')
                    return
                clear_output()

                moves_made = self.move_castle(player_start_pos, player_end_pos, prev_board, use_robot=False)
                if not moves_made:
                    if not self.make_move(player_start_pos, player_end_pos):
                        print('Invalid move')
                        continue

                # -- Run perception system to get pcds-- #
                print('Getting robot move')
                pcds, pieces = self.get_processed_pcds()
                location_to_piece = self.extract_piece_locations(pcds, pieces)

                current_board = [['  ' for i in range(8)] for j in range(8)]
                for coord, piece in location_to_piece:
                    loc = self.board.coord_to_index(coord)
                    current_board[loc[0]][loc[1]] = piece
                    # res[7-loc[1]][loc[0]] = piece

                # -- Plot perception system -- #
                print('-'*23)
                print('Robot understanding of board:')
                Board.print_board(current_board)
                print('-'*23)

                fig = multiplot(pcds)
                fig_path = osp.join(get_chessbot_src(), 'demos/game_viz.html')
                fig.write_html(fig_path)
                iplot(fig)

                self.show_mask_labeled_image()


                # -- Derive player move from perception -- #
                move_correct = False

                # TODO: Update this to use castling move
                derived_player_move = Board.get_move(prev_board, current_board)

                print(f'Actual move: {player_start_pos} -> {player_end_pos}')
                print(f'Predicted move: {derived_player_move[0]} -> {derived_player_move[1]}')

                der_player_start_pos_str = Board.index_to_location(derived_player_move[0])
                der_player_end_pos_str = Board.index_to_location(derived_player_move[1])
                der_player_move = der_player_start_pos_str + der_player_end_pos_str
                move_correct = self.stockfish.is_move_correct(der_player_move)
                if not move_correct:
                    print(f'{der_player_move} is not correct, try again!')
                    # Reset piece to orinal location
                    self.make_move(player_end_pos, player_start_pos)

            # Update internal boards
            self.stockfish.make_moves_from_current_position([der_player_move])
            prev_board[derived_player_move[1][0]][derived_player_move[1][1]] = prev_board[derived_player_move[0][0]][derived_player_move[0][1]]
            prev_board[derived_player_move[0][0]][derived_player_move[0][1]] = '  '


            robot_move = self.stockfish.get_best_move()
            if robot_move is None:
                print('Human wins!')
                ori_z = self.robot_status.get_pose_value('z')
                ori_y = self.robot_status.get_pose_value('y')
                self.robot_status.set_pose_value(z=0.1, y=-0.1)
                self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)
                print("(But I'll win the next one)")
                self.robot_status.set_pose_value(z=ori_z, y=ori_y)
                self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)
                return
            self.stockfish.make_moves_from_current_position([robot_move])

            robot_start_pos_str = robot_move[:2]
            robot_end_pos_str = robot_move[2:]
            robot_start_pos = Board.location_to_coord(robot_start_pos_str)
            robot_end_pos = Board.location_to_coord(robot_end_pos_str)

            print(f'Robot move: {robot_start_pos} -> {robot_end_pos}')

            moves_made = self.move_castle(robot_start_pos, robot_end_pos, prev_board, use_robot=True, pcds=pcds)
            if not moves_made:
                moves_made = [(robot_start_pos, robot_end_pos)]
                self.make_move_with_robot(robot_start_pos, robot_end_pos, pcds)

            for robot_start_pos, robot_end_pos in moves_made:
                prev_board[robot_end_pos[0]][robot_end_pos[1]] = prev_board[robot_start_pos[0]][robot_start_pos[1]]
                prev_board[robot_start_pos[0]][robot_start_pos[1]] = '  '

    def auto_play_game(self, user_stockfish: Stockfish, pre_move_list: list[str]=[]):
        # castle using the king coordinates

        prev_board = deepcopy(Board.starting_board_list)
        self.robot_status.set_pose_value(z=0.4)

        if pre_move_list:
            print('Setting from initial state!')
        for move in pre_move_list:
            start_pos_str = move[:2]
            end_pos_str = move[2:]
            start_pos = Board.location_to_coord(start_pos_str)
            end_pos = Board.location_to_coord(end_pos_str)

            # Check if we're castling
            moves_made = self.move_castle(start_pos, end_pos, prev_board, use_robot=False)

            if not moves_made:
                self.make_move(start_pos, end_pos)
                moves_made = [(start_pos, end_pos)]


            for _start_pos, _end_pos in moves_made:
                prev_board[_end_pos[0]][_end_pos[1]] = prev_board[_start_pos[0]][_start_pos[1]]
                prev_board[_start_pos[0]][_start_pos[1]] = '  '

            self.stockfish.make_moves_from_current_position([move])
            if user_stockfish:
                user_stockfish.make_moves_from_current_position([move])

        if pre_move_list:
            print('Done setting board!')

        turns = 0
        perception_errors = 0
        other_errors = 0
        try:
            while True:
                move_correct = False
                while not move_correct:
                    # Get player move (Player is always white)
                    try:
                        if user_stockfish:
                            player_move = user_stockfish.get_best_move()
                            if player_move is None:  # Robot won
                                # print(f'Errors: {errors} | Turns: {turns} | Completed: {True}')
                                return perception_errors, other_errors, turns, True  # Completed successfully
                            user_stockfish.make_moves_from_current_position([player_move])
                            player_start_pos_str = player_move[:2]
                            player_end_pos_str = player_move[2:]
                            player_start_pos = Board.location_to_coord(player_start_pos_str)
                            player_end_pos = Board.location_to_coord(player_end_pos_str)
                        else:
                            player_start_pos, player_end_pos = self._get_player_move()
                    except ValueError:
                        print('Exiting!')
                        return
                    clear_output()

                    moves_made = self.move_castle(player_start_pos, player_end_pos, prev_board, use_robot=False)
                    if not moves_made:
                        if not self.make_move(player_start_pos, player_end_pos):
                            print('Invalid move')
                            continue

                    # -- Run perception system to get pcds-- #
                    print('Getting robot move')
                    pcds, pieces = self.get_processed_pcds()
                    location_to_piece = self.extract_piece_locations(pcds, pieces)

                    current_board = [['  ' for i in range(8)] for j in range(8)]
                    for coord, piece in location_to_piece:
                        loc = self.board.coord_to_index(coord)
                        current_board[loc[0]][loc[1]] = piece
                        # res[7-loc[1]][loc[0]] = piece

                    # -- Plot perception system -- #
                    print('-'*23)
                    print('Robot understanding of board:')
                    Board.print_board(current_board)
                    print('-'*23)

                    # fig = multiplot(pcds)
                    # fig_path = osp.join(get_chessbot_src(), 'demos/game_viz.html')
                    # fig.write_html(fig_path)
                    # iplot(fig)

                    # self.show_mask_labeled_image()


                    # -- Derive player move from perception -- #
                    move_correct = False

                    # TODO: Update this to use castling move
                    derived_player_move = Board.get_move(prev_board, current_board)

                    print(f'Actual move: {player_start_pos} -> {player_end_pos}')
                    print(f'Predicted move: {derived_player_move[0]} -> {derived_player_move[1]}')
                    # Catch error and correct it.
                    if player_start_pos != derived_player_move[0] or player_end_pos != derived_player_move[1]:
                        perception_errors += 1
                        corrected_player_move = (player_start_pos, player_end_pos)
                        derived_player_move = corrected_player_move


                    der_player_start_pos_str = Board.index_to_location(derived_player_move[0])
                    der_player_end_pos_str = Board.index_to_location(derived_player_move[1])
                    der_player_move = der_player_start_pos_str + der_player_end_pos_str
                    move_correct = self.stockfish.is_move_correct(der_player_move)
                    if not move_correct:
                        print(f'{der_player_move} is not correct, try again!')
                        # Reset piece to orinal location
                        self.make_move(player_end_pos, player_start_pos)

                # Update internal boards
                self.stockfish.make_moves_from_current_position([der_player_move])
                prev_board[derived_player_move[1][0]][derived_player_move[1][1]] = prev_board[derived_player_move[0][0]][derived_player_move[0][1]]
                prev_board[derived_player_move[0][0]][derived_player_move[0][1]] = '  '


                robot_move = self.stockfish.get_best_move()
                if robot_move is None:
                    print('Human wins!')
                    ori_z = self.robot_status.get_pose_value('z')
                    ori_y = self.robot_status.get_pose_value('y')
                    self.robot_status.set_pose_value(z=0.1, y=-0.1)
                    self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)
                    print("(But I'll win the next one)")
                    self.robot_status.set_pose_value(z=ori_z, y=ori_y)
                    self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)
                    turns += 1
                    # print(f'Errors: {errors} | Turns: {turns} | Completed: {True}')
                    return perception_errors, other_errors, turns, True  # Completed successfully

                self.stockfish.make_moves_from_current_position([robot_move])
                user_stockfish.make_moves_from_current_position([robot_move])

                robot_start_pos_str = robot_move[:2]
                robot_end_pos_str = robot_move[2:]
                robot_start_pos = Board.location_to_coord(robot_start_pos_str)
                robot_end_pos = Board.location_to_coord(robot_end_pos_str)

                print(f'Robot move: {robot_start_pos} -> {robot_end_pos}')

                moves_made = self.move_castle(robot_start_pos, robot_end_pos, prev_board, use_robot=True, pcds=pcds)
                if not moves_made:
                    moves_made = [(robot_start_pos, robot_end_pos)]
                    _, corrected = self.make_move_with_robot(robot_start_pos, robot_end_pos, pcds, apply_correction=True)
                    perception_errors += corrected

                    # other_errors += self.correct_robot_move()

                for robot_start_pos, robot_end_pos in moves_made:
                    prev_board[robot_end_pos[0]][robot_end_pos[1]] = prev_board[robot_start_pos[0]][robot_start_pos[1]]
                    prev_board[robot_start_pos[0]][robot_start_pos[1]] = '  '

                turns += 1

        except:
            turns += 1
            other_errors += 1
            # print(f'Errors: {errors} | Turns: {turns} | Completed: {False}')
            return perception_errors, other_errors, turns, False  # Ran into error


    def move_castle(self, start_pos, end_pos, prev_board, use_robot, pcds=None):
        moves_to_make = []
        e1 = (4, 0)
        c1 = (2, 0)
        g1 = (6, 0)

        e8 = (4, 7)
        c8 = (2, 7)
        g8 = (6, 7)

        if start_pos == e1 and end_pos == c1:
            moves_to_make = ['e1c1', 'a1d1']
        elif start_pos == e1 and end_pos == g1:
            moves_to_make = ['e1g1', 'h1f1']
        elif start_pos == e8 and end_pos == c8:
            moves_to_make = ['e8c8', 'e8d8']
        elif start_pos == e8 and end_pos == g8:
            moves_to_make = ['e8g8', 'h8f8']

        coord_moves_to_make = []
        for i, move in enumerate(moves_to_make):
            start_pos_str = move[:2]
            end_pos_str = move[2:]
            start_pos = Board.location_to_coord(start_pos_str)
            end_pos = Board.location_to_coord(end_pos_str)
            coord_moves_to_make.append((start_pos, end_pos))
            if use_robot:
                self.make_move_with_robot(start_pos, end_pos, pcds)
            else:
                self.make_move(start_pos, end_pos)

        return coord_moves_to_make

    def _get_player_move(self):
        player_start_pos = input('Enter start move as (x, y)')
        player_end_pos = input('Enter end move as (x, y)')

        player_start_pos = player_start_pos.strip('()')
        player_start_pos = tuple([int(i) for i in player_start_pos.split(',')])

        player_end_pos = player_end_pos.strip('()')
        player_end_pos = tuple([int(i) for i in player_end_pos.split(',')])

        return player_start_pos, player_end_pos

    def extract_piece_locations(self, pcds: list[np.ndarray], pieces: list[str]) -> list:
        """
        Get piece locations from raw pointclouds.  list of (location, piece) pairs

        Args:
            pcds (list(np.ndarray)): list of pcds in N x 3 format
            pieces (list(str)): list of predicted piece types

        Returns:
            list: (location, piece) tuples where location is (x, y) position in meters
        """
        res = []
        for i, pcd in enumerate(pcds):
            mean_loc = list(np.mean(pcd[:, :2], axis=0))
            res.append((mean_loc, pieces[i]))
        return res

    def run_teleop(self):
        self.meshcat.AddButton("Stop Simulation", "Escape")
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2.0)

        sim_context = self.simulator.get_context()


        self.meshcat.DeleteButton("Stop Simulation")
        self.station.Publish(self.station_context)

    def read_rgbd_sensor(self):
        """
        Get color, depth, and label images for the board

        Returns:
            tuple(array): color image, depth image, label image
        """
        color_image = self.station.GetOutputPort("camera_rgb_image").Eval(self.station_context).data
        depth_image = self.station.GetOutputPort("camera_depth_image").Eval(self.station_context).data
        label_image = self.station.GetOutputPort("camera_depth_image").Eval(self.station_context).data
        return color_image, depth_image, label_image

    def show_label_image(self):
        """
        Call this in a ipynb window.  Shows the output of the mask camera image.
        """
        label_image = self.station.GetOutputPort("camera_label_image").Eval(self.station_context)
        plt.imshow(colorize_labels(label_image.data))
        print('Num Unique values: ', len(np.unique(label_image.data)))

    def show_mask_labeled_image(self):

        pieces = [
            'BB', # : 'Bishop_B.urdf',
            'BW', # : 'Bishop_W.urdf',

            'KB', # : 'King_B.urdf',
            'KW', # : 'King_W.urdf',

            'NB', # : 'Knight_B.urdf',
            'NW', # : 'Knight_W.urdf',

            'PB', # : 'Pawn_B.urdf',
            'PW', # : 'Pawn_W.urdf',

            'QB', # : 'Queen_B.urdf',
            'QW', # : 'Queen_W.urdf',

            'RB', # : 'Rook_B.urdf',
            'RW', # : 'Rook_W.urdf'
        ]

        prediction, img = self.station.GetOutputPort('camera_raw_prediction').Eval(self.mut_station_context)
        thresh = 0.97
        img_np = np.array(img)
        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img_np)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 60)]

        num_instances = prediction[0]['boxes'].shape[0]
        bbox_colors = random.sample(colors, num_instances)
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().detach().numpy()

        for i in range(num_instances):
            if scores[i] < thresh:
                continue
            color = bbox_colors[i]
            bb = boxes[i,:]
            bbox = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                    linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(bb[0], bb[1], pieces[labels[i] - 1],
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
            plt.text(bb[0], bb[3], s=str(f'{scores[i]:.3}'),
                    color='white', verticalalignment='bottom',
                    bbox={'color': color, 'pad': 0})

        plt.axis('off');
        plt.show()


    def show_rgb_image(self):
        """
        Call this in a ipynb window.  Shows the output of the rgb camera image.
        """
        # color_image = self.station.GetOutputPort("camera_rgb_image").Eval(self.station_context)
        color_image = self.station.GetOutputPort("camera_rgb_image").Eval(self.mut_station_context)
        plt.imshow(color_image.data)


    def get_processed_pcds(self):
        """
        Returns list of processed point clouds of pieces, and piece labels
        """
        return self.station.GetOutputPort('icp_pcd_stack').Eval(self.mut_station_context)

    def make_move(self, start_loc: tuple[int], end_loc: tuple[int]) -> bool:
        """
        Move piece at start_loc (0-indexed) to location end_loc (0-indexed).

        Args:
            start_loc (tuple[int]): tuple of (x, y) location 0-indexed
            end_loc (tuple[int]): tuple of (x, y) location 0-indexed

        Returns:
            bool: True if piece was moved, False otherwise
        """
        made_move = False
        for idx in self.idx_to_location.keys():
            piece = self.plant.GetBodyByName("piece_body", idx)
            pose = self.plant.GetFreeBodyPose(self.mut_plant_context,
                piece)

            xyz = pose.translation()
            if self.board.coord_to_index((xyz[0], xyz[1])) == start_loc:
                new_x, new_y = self.board.get_xy_location_from_idx(end_loc[0], end_loc[1])
                new_xyz = np.array([new_x, new_y, xyz[-1]])
                new_pose = RigidTransform(pose.rotation(), new_xyz)
                self.plant.SetFreeBodyPose(self.mut_plant_context,
                    piece, new_pose)
                print('Set pose!')
                made_move = True

            if self.board.coord_to_index((xyz[0], xyz[1])) == end_loc:
                new_x, new_y = 1, 1
                new_xyz = np.array([new_x, new_y, xyz[-1]])
                new_pose = RigidTransform(pose.rotation(), new_xyz)
                self.plant.SetFreeBodyPose(self.mut_plant_context,
                    piece, new_pose)

        if not made_move:
            print('Could not find piece at location!')
            return False
        # self.robot_status.set_gripper_status('closed')
        # print('Robot value: ', self.robot_status.value)
        # self.robot_status.set_pose_value(x = self.robot_status.get_pose_value('x') + 0.2)
        # print('Robot value: ', self.robot_status.value)
        # self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)
        # self.robot_status.set_gripper_status('open')
        # self.robot_status.set_pose_value(x = self.robot_status.get_pose_value('x') - 0.2)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)
        return True

    def make_move_with_robot(self, start_loc: tuple[int], end_loc: tuple[int],
        pcds, apply_correction=False) -> bool:
        """
        Move piece at start_loc (0-indexed) to location end_loc (0-indexed).

        Args:
            start_loc (tuple[int]): tuple of (x, y) location 0-indexed
            end_loc (tuple[int]): tuple of (x, y) location 0-indexed

        Returns:
            bool: True if piece was moved, False otherwise
        """
        corrected = False
        if self.grasp_yaw is None:
            self.grasp_yaw = self.robot_status.get_pose_value('yaw') + np.pi/4
        home_x, home_y, home_z = 0, 0, 0.4

        dump_x, dump_y = 0.4, 0

        clear_z = 0.15
        grab_z_offset = -0.03
        place_z_offset = -0.02

        grasp_locations = self._pcds_to_grasp_locations(pcds)

        board_spacing = self.board.board_spacing
        start_grasp_coords = None
        end_grasp_coords = None
        for grasp_location in grasp_locations:
            grasp_location_idx = self.board.coord_to_index(grasp_location[:2])

            # Use ground truth location if pcd is bad.  For evaluation only (so game can continue if perception is bad)
            if apply_correction:
                grasp_location, move_corrected = self.correct_grasp_location(grasp_location_idx, grasp_location)

            if grasp_location_idx == start_loc:
                if move_corrected:
                    corrected = True
                start_grasp_coords = grasp_location

            if grasp_location_idx == end_loc:
                if move_corrected:
                    corrected = True
                end_grasp_coords = grasp_location

        print(f'Corrected the move: {corrected}')


        place_x, place_y = self.board.get_xy_location_from_idx(end_loc[0], end_loc[1])
        ref_grasp_x, ref_grasp_y = self.board.get_xy_location_from_idx(start_loc[0], start_loc[1])

        # We have a piece in the place we're trying to move to.
        # So we drop piece off the board
        if end_grasp_coords is not None:
            print('Target grasp location: ', end_grasp_coords)
            self.robot_status.set_gripper_status('open')
            self.robot_status.set_pose_value(x = end_grasp_coords[0], y = end_grasp_coords[1], z = clear_z, yaw=self.grasp_yaw)
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)

            self.robot_status.set_pose_value(x = end_grasp_coords[0], y = end_grasp_coords[1], z = grab_z_offset + end_grasp_coords[2])
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)

            # Print grasp location
            link_8_index = self.plant.GetBodyByName("panda_link8").index()
            print('Estimated state: ', self.station.GetOutputPort('body_poses').Eval(self.mut_station_context)[link_8_index].translation())

            self.robot_status.set_gripper_status('close')
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

            self.robot_status.set_pose_value(x = end_grasp_coords[0], y = end_grasp_coords[1], z = clear_z)
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

            self.robot_status.set_pose_value(x = dump_x, y = dump_y, z = clear_z)
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)

            self.robot_status.set_gripper_status('open')
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

            self.robot_status.set_pose_value(x = home_x, y = home_y, z = home_z)
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 3)


        print('Target grasp location: ', start_grasp_coords)
        print('non-adaptive grasp location: ', ref_grasp_x, ref_grasp_y)
        self.robot_status.set_gripper_status('open')
        self.robot_status.set_pose_value(x = start_grasp_coords[0], y = start_grasp_coords[1], z = clear_z, yaw=self.grasp_yaw)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)

        self.robot_status.set_pose_value(x = start_grasp_coords[0], y = start_grasp_coords[1], z = grab_z_offset + start_grasp_coords[2])
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 5)

        # Print grasp location
        link_8_index = self.plant.GetBodyByName("panda_link8").index()
        print('Estimated state: ', self.station.GetOutputPort('body_poses').Eval(self.mut_station_context)[link_8_index].translation())

        self.robot_status.set_gripper_status('close')
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

        self.robot_status.set_pose_value(x = start_grasp_coords[0], y = start_grasp_coords[1], z = clear_z)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

        self.robot_status.set_pose_value(x = place_x, y = place_y, z = clear_z)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2)

        self.robot_status.set_pose_value(x = place_x, y = place_y, z = place_z_offset + start_grasp_coords[2])
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 3)

        self.robot_status.set_gripper_status('open')
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

        self.robot_status.set_pose_value(x = place_x, y = place_y, z = clear_z)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1)

        self.robot_status.set_pose_value(x = home_x, y = home_y, z = home_z, yaw=self.grasp_yaw)
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 3)

        return True, corrected

    def correct_grasp_location(self, loc, derived_grasp_coords) -> int:
        """
        Helper method for evaluating the robot.  Will correct an infered grasp
        location if it is incorrect.

        Returns:
            tuple: new grasp coordinates and true if corrected (else false)
        """
        eps = 0.01
        res_grasp_coords = list(derived_grasp_coords)
        corrected = False
        for idx in self.idx_to_location.keys():
            piece = self.plant.GetBodyByName("piece_body", idx)
            pose = self.plant.GetFreeBodyPose(self.mut_plant_context,
                piece)

            xyz = pose.translation()
            if self.board.coord_to_index((xyz[0], xyz[1])) == loc:
                if abs(derived_grasp_coords[0] - xyz[0]) > eps:
                    res_grasp_coords[0] = xyz[0]
                    corrected = True
                if abs(derived_grasp_coords[1] - xyz[1]) > eps:
                    res_grasp_coords[1] = xyz[1]
                    corrected = True
        return res_grasp_coords, corrected

    def _pcds_to_grasp_locations(self, pcds: list[np.ndarray]) -> list[np.ndarray]:
        grasp_locations = []
        for pcd in pcds:
            grasp_loc = np.mean(pcd, axis=0)
            grasp_loc[-1] *= 1 # Try to grab at the mean
            grasp_locations.append(grasp_loc)

        return grasp_locations
