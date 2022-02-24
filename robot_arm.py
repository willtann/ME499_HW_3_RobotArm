import numpy as np


class RobotArm:

    def __init__(self, *arm_lengths, obstacles=None):
        """
        Represents an N-link arm with the arm lengths given.
        Example of initializing a 3-link robot with a single obstacle:

        my_arm = RobotArm(0.58, 0.14, 0.43, obstacles=[VerticalWall(0.45)])

        :param arm_lengths: Float values representing arm lengths of the robot.
        :param obstacles:
        """
        self.arm_lengths = np.array(arm_lengths)
        if np.any(self.arm_lengths < 0):
            raise ValueError("Cannot have negative arm length!")
        self.obstacles = []
        if obstacles is not None:
            self.obstacles = obstacles

    def __repr__(self):
        msg = '<RobotArm with {} links\nArm lengths: '.format(len(self.arm_lengths))
        msg += ', '.join(['{:.2f}'.format(length) for length in self.arm_lengths])
        msg += '\nObstacles: '
        if not len(self.obstacles):
            msg += 'None'
        else:
            msg += '\n\t' + '\n\t'.join(str(obstacle) for obstacle in self.obstacles)
        msg += '\n>'
        return msg

    def __str__(self):
        return self.__repr__()

    def get_links(self, thetas):
        """
        Returns all of the link locations of the robot as Link objects.
        :param thetas: A list or array of scalars matching the number of arms.
        :return: A list of Link objects.
        """

        cum_theta = np.cumsum(thetas)

        results = np.zeros((self.arm_lengths.shape[0] + 1, 2))

        results[1:, 0] = np.cumsum(self.arm_lengths * np.cos(cum_theta))
        results[1:, 1] = np.cumsum(self.arm_lengths * np.sin(cum_theta))
        links = [Link(start, end) for start, end in zip(results[:-1], results[1:])]

        return links

    def get_ee_location(self, thetas):
        """
        Returns the location of the end effector as a length 2 Numpy array.
        :param thetas: A list or array of scalars matching the number of arms.
        :return: A length 2 Numpy array of the x,y coordinate.
        """
        return self.get_links(thetas)[-1].end

    def ik_grid_search(self, target, intervals):
        raise NotImplementedError

    def ik_fmin_search(self, target, thetas_guess, max_calls=100):
        raise NotImplementedError

    def get_collision_score(self, thetas):
        lin = self.get_links(thetas)
        obs = self.obstacles
        print(obs)
        count = 0
        for i, link in enumerate(lin):
            this_link = Link(link.start, link.end)
            for w in enumerate(obs):
                this_link.check_wall_collision(obs[w])
            count -= this_link
        return count

    def ik_constrained_search(self, target, thetas_guess, max_iters=100):
        raise NotImplementedError

    def plot_robot_state(self, thetas, target=None, filename='robot_arm_state.png'):
        raise NotImplementedError


class Link:

    def __init__(self, start, end):
        """
        Represents a finite line segment in the XY plane, with start and ends given as 2-vectors
        :param start: A length 2 Numpy array
        :param end: A length 2 Numpy array
        """
        self.start = start
        self.end = end

    def __repr__(self):
        return '<Link: ({:.3f}, {:.3f}) to ({:.3f}, {:.3f})>'\
               .format(self.start[0], self.start[1], self.end[0], self.end[1])

    def __str__(self):
        return self.__repr__()

    def check_wall_collision(self, wall):
        if not isinstance(wall, VerticalWall):
            raise ValueError('Please input a valid Wall object to check for collision.')
        else:
            if self.start[0] < wall.loc < self.end[0]:  # x_link_START < x_wall < x_link_END
                # Then the wall is intersecting the link
                return True
            else:
                return False


class VerticalWall:

    def __init__(self, loc):
        """
        A VerticalWall represents a vertical line in space in the XY plane, of the form x = loc.
        :param loc: A scalar value
        """
        self.loc = loc

    def __repr__(self):
        return '<VerticalWall at x={:.3f}>'.format(self.loc)


if __name__ == '__main__':
    # Example of initializing a 3-link robot arm
    # arm = RobotArm(1.2, 0.8, 0.5, obstacles=[VerticalWall(1.2)])
    # print('Robot Arm', arm)

    # Get the end-effector position of the arm for a given configuration
    # thetas = [np.pi / 4, np.pi / 2, -np.pi / 4]
    # thetas = [np.pi / 2, 0, 0]  # Problem 1.2.1
    # thetas = [0, 0, np.pi]  # Problem 1.2.2
    # pos = arm.get_ee_location(thetas)
    # print('End effector is at: ({:.3f}, {:.3f})'.format(*pos))

    # Get each of the links for the robot arm, and print their start and end points
    # links = arm.get_links(thetas)
    #
    # for i, link in enumerate(links):
    #     print('Link {}:'.format(i))
    #     print('\tStart: ({:.3f}, {:.3f})'.format(*link.start))
    #     print('\tEnd: ({:.3f}, {:.3f})'.format(*link.end))

    # Problem 1.1
    print('__________Problem 1.1__________')
    my_link = Link((1.1, 5.0), (3.0, 3.3))
    print('my_link (', my_link.start[0], ')(', my_link.end[0], ')')
    print('Vertical wall at x=2.1... collision = ', my_link.check_wall_collision(VerticalWall(2.1)))
    print('Vertical wall at x=-0.3... collision = ', my_link.check_wall_collision(VerticalWall(-0.3)))

    # Problem 1.2
    print('__________Problem 1.2__________')
    my_arm = RobotArm(1, 1, 1, obstacles=[VerticalWall(1.5), VerticalWall(1.5)])
    my_arm.get_collision_score([np.pi/2, 0, 0])
    # my_arm.get_collision_score([0, 0, np.pi])




