#!/usr/bin/env python



# A very basic TurtleBot script that moves TurtleBot forward indefinitely. Press CTRL + C to stop.  To run:
# On TurtleBot:
# roslaunch turtlebot_bringup minimal.launch
# On work station:
# python goforward.py

import time
import rospy
import math
import pandas as pd
import numpy as np
from geometry_msgs.msg import Twist,PoseStamped
from nav_msgs.msg import Odometry,Path
from array import *

global x,y,path_record

class Test1():
    def __init__(self):
        global path_record
        #global x,y
        # initiliaze
        rospy.init_node('Test1', anonymous=False)
	# tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

	# Create a publisher which can "talk" to TurtleBot and tell it to move
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #self.path_pub = rospy.Publisher('trajectory', Path, queue_size=10)
        #path_record = Path()
        rospy.Subscriber('/odom',Odometry,self.callback)


	#TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10)

        # Twist is a datatype for velocity
        move_cmd = Twist()
	# let's go forward at 0.2 m/s
        #move_cmd.linear.x = 0.20
	# let's turn at 0 radians/s
        #move_cmd.angular.z = 0

        #csvdata = open(r'/home/parallels/helloworld/turtlebot/Trajectory 2/Controller 0 velocity trajectory/controller 0 velocity trajectory 1.csv')
        #data = pd.DataFrame(df)
        #controlist = np.loadtxt(csvdata,delimiter = ',')
        controlist = [[0.16836806, -0.09762034], [0.04734952, -0.09360345], [-0.030327685, -0.09434827], [-0.0769276, -0.09925614], [-0.094307184, -0.105160154], [-0.10871135, -0.11202941], [-0.11501221, -0.115402326], [-0.13449146, -0.1179709], [-0.13393034, -0.12392834], [-0.1426897, -0.120149404], [-0.120742396, -0.1202434], [-0.10458504, -0.11812127], [-0.10176644, -0.12019608], [-0.102271736, -0.12203307], [-0.10332517, -0.119554795], [-0.103348166, -0.11110181], [-0.047489278, -0.123576894], [-0.0018793419, -0.10836193], [-0.07666069, -0.07325221], [-0.10858719, -0.06131251], [-0.041988276, -0.017577976], [-0.01427421, -0.0073633194], [-0.010054067, -0.0059220195], [-0.0072715953, -0.0044833273], [-0.005330898, -0.0033210665]]
        


        #counter = 0
        #interv = 10  #maximum time
        #wait = 15    #waiting time
        con_counter = 0
	# as long as you haven't ctrl + c keeping doing...
        start = time.time()
        while 1:
            
            move_cmd.linear.x = 0.25  #线速度
            move_cmd.angular.z = 0.01 #角速度   跑1s 2s
            self.cmd_vel.publish(move_cmd)
            
            r.sleep()


    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
	# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
	# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)
    def callback(self,msg):
        global x,y,path_record
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        #if 3.4>= x>=3.31:
          #print("end by command")
          #quit() 
        #position_x = []
        #position_x.append(x)
        #print(position_x)
        #for i in position_x:
          #position_x.append(x)
        
        #q1 = data.pose.pose.orientation.x
        #q2 = data.pose.pose.orientation.y
        #q3 = data.pose.pose.orientation.z
        #q4 = data.pose.pose.orientation.w
        #print(msg.pose.pose) ##忘记怎么写
        # initial_position = pd.read_csv(r'/home/parallels/helloworld/turtlebot/Trajectory 2/Controller 1 velocity trajectory/controller 1 initial points.csv')
        # initial_x,initial_y,theta = initial_position.iloc[0]
        # #theta = math.degrees(theta)
        # #initial_x = 2.99606412
        # #initial_y = 2.9978848
        # #theta = 3.92950583
        
        # theta = math.degrees(theta)
        # position_x_change = x*math.cos(-math.radians(theta)) + y*math.sin(-math.radians(theta)) + initial_x
        # position_y_change = y*math.cos(-math.radians(theta)) - x*math.sin(-math.radians(theta)) + initial_y
        # with open("Trajectory_controller_0.csv", "a") as f:
        #         print(position_x_change,",",position_y_change, file=f)
        # f.close()
        # print("our x location is ",position_x_change, " our y location is ", position_y_change)
        #current_time = rospy.Time.now()
        #pose = PoseStamped()
        #pose.header.stamp = current_time
        #pose.header.frame_id = 'odom'
        #pose.pose = msg.pose.pose
        #path_record.header.stamp = current_time
        #path_record.header.frame_id = 'odom'
        #path_record.poses.append(pose)
        #self.path_pub.publish(path_record)
        
        
        

if __name__ == '__main__':
    #try:
        Test1()
        #dist = math.sqrt(x**2 + y**2)
        #print(dist)
        if 5>= x>=4.9 and 5>= y >= 4.9:
            print("robot arrivied")
        else:
            print("not arrivied")
        quit()
            

    #except:
        #rospy.loginfo("GoForward node terminated.")