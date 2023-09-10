import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from mann_pytorch.GatingNetwork import GatingNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from mann_pytorch.MotionPredictionNetwork import MotionPredictionNetwork
from gym_ignition.rbd.idyntree import kindyncomputations

from typing import List
import numpy as np
from gym_ignition.rbd.idyntree import numpy


class MANN(nn.Module):
    """Class for the Mode-Adaptive Neural Network."""

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 num_experts: int, gn_hidden_size: int, mpn_hidden_size: int, dropout_probability: float,
                 kindyn: kindyncomputations.KinDynComputations):
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
        input_size = 136 #only the first 136 elements correspond to the original training features
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

        # Store the kindyn object
        self.kindyn = kindyn

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['kindyn']
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
    
    def reset_robot_configuration(self, joint_positions: List, base_position: List, base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = numpy.FromNumPy.to_idyntree_transform(
            position=np.array(base_position),
            quaternion=np.array(base_quaternion)).asHomogeneousTransform().toNumPy()

        self.kindyn.set_robot_state(s=joint_positions, ds=np.zeros(len(joint_positions)), world_H_base=world_H_base)

    def get_pi_loss_components(self, X: torch.Tensor, pred: torch.Tensor) -> (torch.Tensor, torch.Tensor):
            
            batch_size = len(X)

            #Get base position and orientation from data for robot state update
            joint_position_batch = X[:,72:104]
            joint_velocity_batch = X[:,104:136]
            base_position_batch = X[:,136:139]
            base_quaternion_batch = X[:,139:]

            # Get Vb from network output
            V_b_linear = pred[:,:3]
            V_b_angular = pred[:,21:24]
            V_b = torch.cat((V_b_linear, V_b_angular), 1)

            V_b_label_array = []

            # Calculate Vb from data for each elem
            for i in range(batch_size):

                # Update robot configuration
                self.reset_robot_configuration(joint_positions=np.array(joint_position_batch[i,:]),
                                           base_position=base_position_batch[i,:],
                                           base_quaternion=base_quaternion_batch[i,:])

                # Get transforms for feet
                world_H_base = self.kindyn.get_world_base_transform()
                base_H_LF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="l_sole")
                W_H_LF = world_H_base.dot(base_H_LF)
                base_H_RF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="r_sole")
                W_H_RF = world_H_base.dot(base_H_RF)

                # Check which foot is lower to determine support foot (gamma=1 for LF support, gamma=0 for RF support)
                if W_H_LF[2,-1] < W_H_RF[2,-1]:
                    gamma = 1
                else:
                    gamma = 0

                # Get foot Jacobians (expressed in base frame)
                lf_jacobian = self.kindyn.get_frame_jacobian("l_sole")

                rf_jacobian = self.kindyn.get_frame_jacobian("r_sole")

                V_b_label = - gamma * torch.matmul(torch.linalg.inv(torch.from_numpy(lf_jacobian[:,:6])), \
                                                  (torch.matmul(torch.from_numpy(lf_jacobian[:,6:]), joint_velocity_batch[i,:]))) \
                            - (1 - gamma) * torch.matmul(torch.linalg.inv(torch.from_numpy(rf_jacobian[:,:6])), \
                                                        (torch.matmul(torch.from_numpy(rf_jacobian[:,6:]), joint_velocity_batch[i,:])))
                V_b_label_array.append(V_b_label)
            
            V_b_label_tensor = torch.stack(V_b_label_array)

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

            pred = self(X[:,:136].float()).double()
            mse_loss = loss_fn(pred, y)

            # Add MSE of Vb and Vbpred
            V_b_label_tensor, V_b = self.get_pi_loss_components(X, pred)
            pi_loss = 10.0 * loss_fn(V_b_label_tensor, V_b)
            
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

    def test_loop(self, loss_fn: _Loss) -> None:
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
                cumulative_test_pi_loss += 10.0 * loss_fn(V_b_label_tensor, V_b).item()

                cumulative_test_loss = cumulative_test_mse_loss + cumulative_test_pi_loss

        # Print the average test loss at the current epoch
        avg_test_loss = cumulative_test_loss/num_batches
        avg_test_mse_loss = cumulative_test_mse_loss/num_batches
        avg_test_pi_loss = cumulative_test_pi_loss/num_batches
        print(f"Avg test loss: {avg_test_loss:>8f} \n")
        print(f"Avg test MSE loss: {avg_test_mse_loss:>8f} \n")
        print(f"Avg test PI loss: {avg_test_pi_loss:>8f} \n")

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

