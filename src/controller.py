#!/usr/bin/env python
import rospy
PKG = 'ilocator_control'
import roslib; roslib.load_manifest(PKG)
import rospkg
from geometry_msgs.msg  import Twist
from turtlesim.msg import Pose
import numpy as np
import os


class ilocatorbot():

	def __init__(self):
	    #Creating our node,publisher and subscriber
	    rospy.init_node('ilocatorbot_controller', anonymous=True)
	    self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
	    self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.callback)
	    self.pose = Pose()
	    self.velocity = 0.3
	    self.lookahead = 0.5
	    self.path = np.genfromtxt(os.path.join(rospkg.RosPack().get_path('ilocator_control'),'data/path.csv'), delimiter = ',')
	    self.rate = rospy.Rate(10)

	#Callback function implementing the pose value received
	def callback(self, data):
	    self.pose = data
	    self.pose.x = round(self.pose.x, 4)
	    self.pose.y = round(self.pose.y, 4)


	def is_alive(self):
		print(self.path)
		rospy.spin()


	def run(self):
		p1 = 0
		p2 = 1
		while p2 < len(self.path):
			robot_pos = [self.pose.x, self.pose.y, self.pose.theta]
			print(robot_pos)	

			start_point = self.path[p1]
			end_point = self.path[p2]
			v,omega = self.purePursuitController(start_point, end_point, self.velocity, robot_pos,self.lookahead)
			vel_msg = Twist()

			#linear velocity in the x-axis:
			vel_msg.linear.x = self.velocity
			vel_msg.linear.y = 0
			vel_msg.linear.z = 0

			#angular velocity in the z-axis:
			vel_msg.angular.x = 0
			vel_msg.angular.y = 0
			vel_msg.angular.z = omega

			#Publishing our vel_msg
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()

			if np.linalg.norm(robot_pos[:2] - self.path[p2]) < 0.1:
				p1 += 1
				p2 += 1
		vel_msg.linear.x = 0
		vel_msg.angular.z = 0
		self.velocity_publisher.publish(vel_msg)
		rospy.spin()


	def purePursuitController(self, p1, p2, velocity, robot_pos, l):	
	    epsilon = 0.0001
	    az = (p1[1]-p2[1])/(p1[0]-p2[0]+epsilon)
	    bz = p1[1]-(p1[1]-p2[1])/(p1[0]-p2[0]+epsilon)*p1[0]
	    dist_from_path = np.abs(az*robot_pos[0]-robot_pos[1]+bz+epsilon)/np.sqrt(az**2+1)
	    dist_to_goal = np.sqrt((robot_pos[0]-p2[0])**2+(robot_pos[1]-p2[1])**2)
	    if dist_from_path > l:
	        ac = -1.0/(az+epsilon)
	        bc = robot_pos[1]+(p1[0]-p2[0])/(p1[1]-p2[1])*robot_pos[0]
	        xc_num = (robot_pos[1]+(p1[0]-p2[0])/(p1[1]-p2[1]+epsilon)*robot_pos[0]-p1[1]+(p1[1]-p2[1])/(p1[0]-p2[0]+epsilon)*p1[0])
	        xc_den = (p1[1]-p2[1])/(p1[0]-p2[0]+epsilon) + (p1[0]-p2[0])/(p1[1]-p2[1]+epsilon)
	        xc = xc_num/xc_den
	        yc = ac*xc + bc
	        pursuit_point = [xc, yc]
	    else:
	        if dist_to_goal <= l:
	            pursuit_point = p2
	            l = dist_to_goal
	        else:
	            coeffs = [1+az**2, 2*az*bz-2*az*robot_pos[1]-2*robot_pos[0], 
	                      bz**2-2*bz*robot_pos[1]-l**2+robot_pos[0]**2+robot_pos[1]**2]
	            roots = np.real(np.roots(coeffs))
	            y = az*roots+bz
	            distance = np.sqrt((roots-p2[0])**2+(y-p2[1])**2)
	            min_root_index = np.argmin(distance)        
	            pursuit_point = [roots[min_root_index],y[min_root_index]]
	    translation_matrix = np.array([[np.cos(robot_pos[2]),-np.sin(robot_pos[2]), robot_pos[0]],
	                          [np.sin(robot_pos[2]), np.cos(robot_pos[2]), robot_pos[1]],
	                          [0, 0, 1]])
	    pursuit_point_matrix = np.expand_dims(np.transpose([pursuit_point[0],pursuit_point[1], 1]),axis=1)
	    robot_coord_matrix = np.matmul(np.linalg.inv(translation_matrix),pursuit_point_matrix)
	    curvature = 2*robot_coord_matrix[1]/l**2
	    omega = velocity*curvature
	    return velocity, omega




if __name__ == '__main__':
	try:
	    #Testing our function
	    x = ilocatorbot()
	    x.run()

	except rospy.ROSInterruptException: pass