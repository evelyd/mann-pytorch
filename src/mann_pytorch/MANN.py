import os
from typing import List, Dict
import json
import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from mann_pytorch.GatingNetwork import GatingNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from mann_pytorch.MotionPredictionNetwork import MotionPredictionNetwork

import adam
from adam.pytorch import KinDynComputationsBatch
from adam.pytorch import KinDynComputations #for ad check

from typing import List

# For Jacobian test
import pathlib
import pytest


class MANN(nn.Module):
    """Class for the Mode-Adaptive Neural Network."""

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 num_experts: int, gn_hidden_size: int, mpn_hidden_size: int, dropout_probability: float,
                 savepath: str,
                 kinDyn: KinDynComputationsBatch):
        """Mode-Adaptive Neural Network constructor.

        Args:
            train_dataloader (DataLoader): Iterable over the training dataset
            test_dataloader (DataLoader): Iterable over the testing dataset
            num_experts (int): The number of expert weights constituting the Motion Prediction Network
            gn_hidden_size (int): The dimension of the 3 hidden layers of the Gating Network
            mpn_hidden_size (int): The dimension of the 3 hidden layers of the Motion Prediction Network
            dropout_probability (float): The probability of an element to be zeroed in the network training
        """

        # Superclass constructor
        super(MANN, self).__init__()

        # Retrieve input and output dimensions from the training dataset
        train_features, train_labels = next(iter(train_dataloader))
        input_size = train_features.size()[-1]
        output_size = train_labels.size()[-1]

        # Define the two subnetworks composing the MANN architecture
        self.gn = GatingNetwork(input_size=input_size,
                                output_size=num_experts,
                                hidden_size=gn_hidden_size,
                                dropout_probability=dropout_probability)
        self.mpn = MotionPredictionNetwork(num_experts=num_experts,
                                           input_size=input_size,
                                           output_size=output_size,
                                           hidden_size=mpn_hidden_size,
                                           dropout_probability=dropout_probability)

        # Store the dataloaders for training and testing
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.savepath = savepath

        # Store the kindyn object
        self.kinDyn = kinDyn

        self.pi_weight = 10.0

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['kinDyn']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mode-Adaptive Neural Network architecture.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            y (torch.Tensor): The output of the Motion Prediction Network
        """

        # Retrieve the output of the Gating Network
        blending_coefficients = self.gn(x.T)

        # Retrieve the output of the Motion Prediction Network
        y = self.mpn(x, blending_coefficients=blending_coefficients)

        return y

    def read_from_file(self, filename: str) -> np.array:
        """Read data as json from file."""

        with open(filename, 'r') as openfile:
            data = json.load(openfile)

        return np.array(data)

    def load_input_mean_and_std(self, datapath: str) -> (List, List):
        """Compute input mean and standard deviation."""

        # Full-input mean and std
        Xmean = self.read_from_file(datapath + 'X_mean.txt')
        Xstd = self.read_from_file(datapath + 'X_std.txt')

        # Remove zeroes from Xstd
        for i in range(Xstd.size):
            if Xstd[i] == 0:
                Xstd[i] = 1

        Xmean = torch.from_numpy(Xmean)
        Xstd = torch.from_numpy(Xstd)

        return Xmean, Xstd

    def load_output_mean_and_std(self, datapath: str) -> (List, List):
        """Compute output mean and standard deviation."""

        # Full-output mean and std
        Ymean = self.read_from_file(datapath + 'Y_mean.txt')
        Ystd = self.read_from_file(datapath + 'Y_std.txt')

        # Remove zeroes from Ystd
        for i in range(Ystd.size):
            if Ystd[i] == 0:
                Ystd[i] = 1

        Ymean = torch.from_numpy(Ymean)
        Ystd = torch.from_numpy(Ystd)

        return Ymean, Ystd

    def denormalize(self, X: torch.Tensor, Xmean: torch.Tensor, Xstd: torch.Tensor) -> torch.Tensor:
        """Denormalize X, given its mean and std."""

        if torch.cuda.is_available:
            Xmean = Xmean.to('cuda')
            Xstd = Xstd.to('cuda')

        # Denormalize
        X = X * Xstd + Xmean

        return X

    def euler_to_quaternion(self, angle):
        """
        Convert Euler angles to quaternion representation.

        Args:
            angle: Tensor containing xyz Euler angles

        Returns:
            quaternions: Tensor containing quaternions in xyzw format.
        """

        roll, pitch, yaw = angle.unbind(dim=-1)

        # print(angle)

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        return torch.stack((qx, qy, qz, qw), dim=-1)

    def quaternion_position_to_transform(self, quaternion: torch.Tensor, position: torch.Tensor):

        # print(quaternion)
        # Normalize quaternion
        quaternion = quaternion / torch.norm(quaternion)

        # Extract individual components of the quaternion
        x, y, z, w = quaternion.unbind(dim=-1)

        # Compute rotation matrix components
        xx = x * x
        xy = x * y
        xz = x * z
        yy = y * y
        yz = y * z
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotation_matrix = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
            ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))

        # Stack rotation matrix with position
        transformation_matrix = torch.cat([
            torch.cat([rotation_matrix, position.unsqueeze(-1)], dim=-1),
            torch.tensor([0, 0, 0, 1], dtype=quaternion.dtype, device=quaternion.device).expand(position.shape[:-1] + (1, 4))
            ], dim=-2)

        return transformation_matrix

    # Compute the plain Jacobian.
    # This function will be used to compute the Jacobian derivative with AD.
    # Given q, computing J̇ by AD-ing this function should work out-of-the-box with
    # all velocity representations, that are handled internally when computing J.
    def J_l_sole(self, q) -> torch.Tensor:

        base_position = q[:3]
        base_quaternion = q[3:7]
        joint_position = q[7:]
        H_b = self.quaternion_position_to_transform(base_quaternion, base_position)

        # Retrieve the robot urdf model
        urdf_path = pathlib.Path("../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

        # Define the joints of interest for the features computation and their associated indexes in the robot joints  list
        controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                            'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                            'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                            'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                            'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                            'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

        # Create a KinDynComputations object with adam
        kinDyn_ad = KinDynComputations(urdf_path, controlled_joints, 'root_link')
        # choose the representation you want to use the body fixed representation
        kinDyn_ad.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)

        #define a new kindyn object for this test only
        full_foot_jacobian = kinDyn_ad.jacobian('l_sole', H_b, joint_position)

        return full_foot_jacobian

    def J_r_sole(self, q) -> torch.Tensor: #not sure i can put the kindyn as an object here boh

        base_position = q[:3]
        base_quaternion = q[3:7]
        joint_position = q[7:]
        H_b = self.quaternion_position_to_transform(base_quaternion, base_position)

        # Retrieve the robot urdf model
        urdf_path = pathlib.Path("../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

        # Define the joints of interest for the features computation and their associated indexes in the robot joints  list
        controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                            'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                            'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                            'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                            'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                            'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

        # Create a KinDynComputations object with adam
        kinDyn_ad = KinDynComputations(urdf_path, controlled_joints, 'root_link')
        # choose the representation you want to use the body fixed representation
        kinDyn_ad.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)

        #define a new kindyn object for this test only
        full_foot_jacobian = kinDyn_ad.jacobian('r_sole', H_b, joint_position)

        return full_foot_jacobian

    def compute_q(self, base_position, base_orientation, joint_position) -> torch.Tensor:

        q = torch.hstack(
            [base_position, base_orientation, joint_position]
        )

        return q

    def Q(self, q: torch.Tensor) -> torch.Tensor:
            # Extract individual components of the quaternion
            qx, qy, qz, qw = q.unbind(dim=-1)

            Q = torch.stack([
            qw, -qx, -qy, -qz,
            qx, qw, -qz, qy,
            qy, qz, qw, -qx,
            qz, -qy, qx, qw
            ], dim=-1).reshape(q.shape[:-1] + (4, 4))

            return Q

    def compute_q̇(self, H_b, base_orientation, V_b, joint_velocity) -> torch.Tensor:
        B_ω_WB = V_b[:,3:6] #have this inside of V_b, already in body fixed

        # need to figure out how to put my linear velocity in mixed repr from body fixed
        W_ṗ_WB = V_b[:, :3]

        # will have to rotate it by {}^W R_B to get mixed repr
        W_ṗ_B = (H_b[:, :3,:3] @ W_ṗ_WB.unsqueeze(-1)).squeeze()

        Q = self.Q(base_orientation)

        dot_prod =  torch.einsum('bi,bi->b', B_ω_WB, B_ω_WB)
        norm_ω = torch.where(dot_prod < (1e-6) ** 2, 1e-6, torch.linalg.norm(B_ω_WB, dim=-1)) #.unsqueeze(-1) #this is dim 32, need to unsqueeze?

        W_Q̇_B = (0.5 * (
            Q
            @ torch.hstack(
                [
                    (0.0 * norm_ω * (1 - torch.linalg.norm(base_orientation, dim=-1))).unsqueeze(-1),
                    B_ω_WB,
                ]
            ).unsqueeze(-1)
        )).squeeze()

        q̇ = torch.hstack([W_ṗ_B, W_Q̇_B, joint_velocity])

        return q̇

    #TODO remove self from passed variables (arg that isn't vectorized)?
    def compute_Vb_label(self, base_position: torch.Tensor, base_angle: torch.Tensor, joint_position: torch.Tensor, V_b: torch.Tensor,
                         joint_velocity: torch.Tensor) -> torch.Tensor:

        # Get a base transform matrix from the data (not prediction) base position and quaternion
        base_quaternion = self.euler_to_quaternion(base_angle)
        H_b = self.quaternion_position_to_transform(base_quaternion, base_position).squeeze()
        full_jacobian_LF = self.kinDyn.jacobian("l_sole", H_b, joint_position)
        full_jacobian_RF = self.kinDyn.jacobian("r_sole", H_b, joint_position)

        # Compute AD Jacobian -------------------------------------------------------
        # Compute q and q̇.
        q = self.compute_q(base_position, base_quaternion, joint_position)
        q̇ = self.compute_q̇(H_b, base_quaternion, V_b, joint_velocity)


        # def my_jac(H, s):
            # return self.kinDyn.jacobian("l_sole", H, s)
        # print("autograd j: ", autograd_j)
        # dJ_dq_LF = torch.func.jacfwd(self.J_sole)(q)
        # dJ_dq_LF = torch.func.jacfwd(self.J, argnums=-1)("l_sole", q) # -1 for q being the last arg but the only one to be differentiated
        # dJ_dq_RF = torch.func.jacfwd(self.J, argnums=-1)("r_sole", q) #

        # print("q̇ size: ", q̇.size())

        # Calculate the Jdot value for each batch element separately
        dJ_dq_LF = torch.zeros(base_position.size()[0], full_jacobian_LF.size()[1], full_jacobian_LF.size()[2], q.size()[1])
        dJ_dq_RF = torch.zeros(base_position.size()[0], full_jacobian_LF.size()[1], full_jacobian_LF.size()[2], q.size()[1])
        # print("empty djdq size: ", dJ_dq_LF.size())
        for b in range(0, base_position.size()[0]):
            # Compute dJ/dt with AD.
            # dJ_dq_LF_b = torch.autograd.functional.jacobian(self.J_l_sole, (q[b,:])) #This is the value that has to be calculated individually
            # dJ_dq_RF_b = torch.autograd.functional.jacobian(self.J_r_sole, (q[b,:]))

            dJ_dq_LF_b = torch.func.jacfwd(self.J_l_sole)(q[b,:])
            dJ_dq_RF_b = torch.func.jacfwd(self.J_r_sole)(q[b,:])
            # print("dJ_dq_LF size: ", dJ_dq_LF.size())
            # input("continue")

            dJ_dq_LF[b,:, :, :] = dJ_dq_LF_b
            dJ_dq_RF[b,:, :, :] = dJ_dq_RF_b
        #TODO stack up all these to create the batch

        #These need to take only the bth element for the calculation
        #first input: batch, 6, 6+ndof, 7+ndof (bijq)
        #second input: batch, 7+ndof (bq)
        # output: batch, 6, 6+ndof (bij)
        # print(dJ_dq_LF)
        # print(q̇)
        q̇ = q̇.unsqueeze(1).unsqueeze(2)
        O_J̇_ad_WL_I_LF = torch.einsum("bmnq,bijq->bmn", dJ_dq_LF.double(), q̇)
        # O_J̇_ad_WL_I_RF = torch.einsum("bijq,bqk->bijk", dJ_dq_RF.double(), q̇.unsqueeze(-1)).squeeze(-1)

        # print("O_J̇_ad_WL_I_LF output: ", O_J̇_ad_WL_I_LF.size())

        # Get the generalized velocity.
        I_ν = torch.hstack([V_b, joint_velocity]).squeeze()

        # Compute J̇.
        O_J̇_WL_I_LF = self.kinDyn.jacobian_dot("l_sole", H_b, joint_position, V_b, joint_velocity)
        # O_J̇_WL_I_RF = self.kinDyn.jacobian_dot("r_sole", H_b, joint_position, V_b, joint_velocity)

        # print("O_J̇_WL_I_LF size: ", O_J̇_WL_I_LF.size())

        # Left foot check
        print(O_J̇_ad_WL_I_LF[0,0,:])
        print(O_J̇_WL_I_LF[0,0,:])
        print((O_J̇_ad_WL_I_LF - O_J̇_WL_I_LF)[0,0,:])
        # print(O_J̇_ad_WL_I_LF[0,:,:,:])
        assert O_J̇_ad_WL_I_LF.cpu().detach().numpy() == pytest.approx(O_J̇_WL_I_LF.cpu().detach().numpy(), abs=2.0)
        # assert torch.einsum("l6g,g->l6", O_J̇_ad_WL_I_LF, I_ν) == pytest.approx(
            # torch.einsum("l6g,g->l6", O_J̇_WL_I_LF, I_ν)
        # )
        # Right foot check
        # assert O_J̇_ad_WL_I_RF.cpu().detach().numpy() == pytest.approx(O_J̇_WL_I_RF.cpu().detach().numpy(), abs=2.0)
        # assert torch.einsum("l6g,g->l6", O_J̇_ad_WL_I_RF, I_ν) == pytest.approx(
            # torch.einsum("l6g,g->l6", O_J̇_WL_I_RF, I_ν)
        # )
        # Compute AD Jacobian -------------------------------------------------------

        # Check which foot is lower to determine support foot (gamma=1 for LF support, gamma=0 for RF support)
        H_LF = self.kinDyn.forward_kinematics("l_sole", H_b, joint_position)
        H_RF = self.kinDyn.forward_kinematics("r_sole", H_b, joint_position)
        z_diff = H_LF[:,2,3] - H_RF[:,2,3]
        condition = z_diff > 0
        gamma = torch.where(condition, 0, 1)
        #so the problem is that the resultant matrix below is not 3 x whatever
        # print(torch.inverse(full_jacobian_LF[:,:,:6]).size())
        # print(full_jacobian_LF[:,:,6:].size())
        # print(joint_velocity.size())
        # print((torch.einsum("bij,bjk->bik", torch.inverse(full_jacobian_LF[:,:,:6]), full_jacobian_LF[:,:,6:])).size())
        # print((torch.einsum("bik,bk->bi", torch.einsum("bij,bjk->bik", torch.inverse(full_jacobian_LF[:,:,:6]), full_jacobian_LF[:,:,6:]).double(), joint_velocity)).size())
        # print(gamma.size())
        # print((torch.inverse(full_jacobian_LF[:,:,:6]) @ full_jacobian_LF[:,:,6:] @ joint_velocity).size())
        # V_b_label = - gamma * torch.inverse(full_jacobian_LF[:,:,:6]) @ full_jacobian_LF[:,:,6:] @ joint_velocity \
                    # - (1 - gamma) * torch.inverse(full_jacobian_RF[:,:,:6]) @ full_jacobian_RF[:,:,6:] @ joint_velocity
        V_b_label = torch.einsum('b,bi->bi', -gamma, torch.einsum("bik,bk->bi", torch.einsum("bij,bjk->bik", torch.inverse(full_jacobian_LF[:,:,:6]), full_jacobian_LF[:,:,6:]).double(), joint_velocity)) \
                    - torch.einsum('b,bi->bi', (1 - gamma), torch.einsum("bik,bk->bi", torch.einsum("bij,bjk->bik", torch.inverse(full_jacobian_RF[:,:,:6]), full_jacobian_RF[:,:,6:]).double(), joint_velocity))

        return V_b_label

    def get_pi_loss_components(self, X: torch.Tensor, pred: torch.Tensor) -> (torch.Tensor, torch.Tensor):

            batch_size = len(X)

            datapath = os.path.join(self.savepath, "normalization/")
            Xmean, Xstd = self.load_input_mean_and_std(datapath)
            Ymean, Ystd = self.load_output_mean_and_std(datapath)

            # Denormalize for correct calculations
            X = self.denormalize(X, Xmean, Xstd)
            pred = self.denormalize(pred, Ymean, Ystd)

            #Get base position and orientation from data for robot state update
            # base_position_batch = X[:,124:127]
            # base_quaternion_batch = X[:,127:]

            # Get Vb from network output
            V_b_linear = pred[:,:3]
            V_b_angular = pred[:,21:24]
            joint_position_batch = pred[:, 42:68]
            joint_velocity_batch = pred[:,68:94]
            base_position_batch = pred[:,-6:-3]
            base_angle_batch = pred[:,-3:]
            V_b = torch.cat((V_b_linear, V_b_angular), 1)

            V_b_label_array = []

            # Calculate Vb from data for each elem
            # batched_V_b_label_fn = torch.vmap(self.compute_Vb_label)
            # V_b_label_tensor = batched_V_b_label_fn(base_position_batch, base_angle_batch, joint_position_batch, V_b, joint_velocity_batch)
            V_b_label_tensor = self.compute_Vb_label(base_position_batch, base_angle_batch, joint_position_batch, V_b, joint_velocity_batch)

            return V_b_label_tensor, V_b

    def train_loop(self, loss_fn: _Loss, optimizer: Optimizer, epoch: int, writer: SummaryWriter) -> None:
        """Run one epoch of training.

        Args:
            loss_fn (_Loss): The loss function used in the training process
            optimizer (Optimizer): The optimizer used in the training process
            epoch (int): The current training epoch
            writer (SummaryWriter): The updater of the event files to be consumed by TensorBoard
        """

        # Total number of batches
        total_batches = int(len(self.train_dataloader))

        # Cumulative loss
        cumulative_loss = 0
        cumulative_mse_loss = 0
        cumulative_pi_loss = 0

        # Print the learning rate and weight decay of the current epoch
        print('Current lr:', optimizer.param_groups[0]['lr'])
        print('Current wd:', optimizer.param_groups[0]['weight_decay'])

        # Iterate over batches
        for batch, (X, y) in enumerate(self.train_dataloader):

            pred = self(X.float()).double()
            mse_loss = loss_fn(pred, y)

            # Add MSE of Vb and Vbpred
            V_b_label_tensor, V_b = self.get_pi_loss_components(X, pred)
            pi_loss = self.pi_weight * loss_fn(V_b_label_tensor, V_b)

            loss = mse_loss + pi_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()
            cumulative_mse_loss += mse_loss.item()
            cumulative_pi_loss += pi_loss.item()

            # Periodically print the current average loss
            if batch % 1000 == 0:
                current_avg_loss = cumulative_loss/(batch+1)
                current_avg_mse_loss = cumulative_mse_loss/(batch+1)
                current_avg_pi_loss = cumulative_pi_loss/(batch+1)
                print(f"avg loss: {current_avg_loss:>7f}  [{batch:>5d}/{total_batches:>5d}]")
                print(f"avg MSE loss: {current_avg_mse_loss:>7f}")
                print(f"avg PI loss: {current_avg_pi_loss:>7f}")

        # Print the average loss of the current epoch
        avg_loss = cumulative_loss/total_batches
        avg_mse_loss = cumulative_mse_loss/total_batches
        avg_pi_loss = cumulative_pi_loss/total_batches
        print("Final avg loss:", avg_loss)
        print("Final avg MSE loss:", avg_mse_loss)
        print("Final avg PI loss:", avg_pi_loss)

        # Store the average loss, loss components, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_loss', avg_loss, epoch)
        writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch)
        writer.add_scalar('avg_pi_loss', avg_pi_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('wd', optimizer.param_groups[0]['weight_decay'], epoch)
        writer.flush()

    def test_loop(self, loss_fn: _Loss, epoch: int, writer: SummaryWriter) -> None:
        """Test the trained model on the test data.

        Args:
            loss_fn (_Loss): The loss function used for testing
        """

        # Dataset dimension
        num_batches = len(self.test_dataloader)

        # Cumulative loss
        cumulative_test_loss = 0
        cumulative_test_mse_loss = 0
        cumulative_test_pi_loss = 0

        with torch.no_grad():

            # Iterate over the testing dataset
            for X, y in self.test_dataloader:

                pred = self(X.float()).double()
                cumulative_test_mse_loss += loss_fn(pred, y).item()

                V_b_label_tensor, V_b = self.get_pi_loss_components(X, pred)
                cumulative_test_pi_loss += self.pi_weight * loss_fn(V_b_label_tensor, V_b).item()

                cumulative_test_loss = cumulative_test_mse_loss + cumulative_test_pi_loss

        # Print the average test loss at the current epoch
        avg_test_loss = cumulative_test_loss/num_batches
        avg_test_mse_loss = cumulative_test_mse_loss/num_batches
        avg_test_pi_loss = cumulative_test_pi_loss/num_batches
        print(f"Avg test loss: {avg_test_loss:>8f} \n")
        print(f"Avg test MSE loss: {avg_test_mse_loss:>8f} \n")
        print(f"Avg test PI loss: {avg_test_pi_loss:>8f} \n")

        # Store the average loss, loss components, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_test_loss', avg_test_loss, epoch)
        writer.add_scalar('avg_test_mse_loss', avg_test_mse_loss, epoch)
        writer.add_scalar('avg_test_pi_loss', avg_test_pi_loss, epoch)
        writer.flush()

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference step on the given input.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            pred (torch.Tensor): The output of the Motion Prediction Network
        """

        with torch.no_grad():
            pred = self(x.float()).double()

        return pred

