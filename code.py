'''
Author: Ashlee Tiwari
Email: ashleetiwari2021@u.northwestern.edu
'''
import numpy as np
import matplotlib.pyplot as plt

class motion_model:
    ''' motion model :calculates new robot state x{t} based on the previous state x_{t-1} and the control inputs {u,v} '''
    def __init__(self,x_start=0,y_start=0,theta_start=0):
        '''
         initializes with robot's starting state 
        '''
        self.x=x_start
        self.y=y_start 
        self.theta=theta_start 

    def estimate_state(self,v,w,delta_t):
        '''     
            Input
                previous state: [x_t-1, y_t-1, θ_t-1]
                control inputs: [v, w]
            Output
                new state estimate :[x_t, y_t, θ_t]
            
            θ_t  =  θ_t-1 + w * delta_t
            x_t  =  x_t-1 + v * delta_t* cos(θ_t-1 + w * delta_t)
            y_t  =  y_t-1 + v * delta_t* sin(θ_t-1 + w * delta_t)
        '''
        self.theta=self.theta+w*(delta_t)
        self.x=self.x+v*np.cos(self.theta)*delta_t
        self.y=self.y+v*np.sin(self.theta)*delta_t         
        return self.x,self.y,self.theta

class measurement_model:
    ''' 
        measurement_model calculates what the range and bearing to the landmark 
        would have been if the estimated estimated mean were correct. 
    '''
    def __init__(self):
        self.r=0
        self.b=0

    def true_landmarks_location(self,j):
        '''  returns the true position of landmark with a particular id 

            Input 
                j : id of landmark 
            Output
                mx: true mean x position of landmark j in meters
                my : true mean y position of landmark j in meters
        '''
        data=np.loadtxt('ds1/ds1_Landmark_Groundtruth.dat',unpack = True)
        all_ids=list(data[0])
        # find the index whose id = j
        ind=all_ids.index(j)
        self.mx=data[1][ind]
        self.my=data[2][ind]
        return self.mx,self.my 
    
    def predicted_landmarks_measurement(self,x,y,theta):
        '''
           Input
             x,y,theta : robot's estimated state
           Output
             r:range to landmark j in meters
             b:bearing to landmark j in radians
           
           range   =  sqrt((landmark_x - x_t)^2 + (landmark_y- y_t)^2)
           bearing  =  atan2((landmark_y - y_t) / (landmark_x  - x_t)) - θ_t

        '''
        dx=self.mx-x
        dy=self.my-y
        self.r=np.sqrt(dx*dx + dy*dy)
        self.b=np.arctan2(dy,dx)- theta
        return self.r,self.b


class ParticleFilter():
    def __init__(self, num_particles, motion_noise, measurement_noise):
        #loading data 
        self.groundtruth_data = np.loadtxt("ds1/ds1_Groundtruth.dat")
        self.landmark_groundtruth_data = np.loadtxt("ds1/ds1_Landmark_Groundtruth.dat")
        self.measurement_data = np.loadtxt("ds1/ds1_Measurement.dat")
        self.odometry_data = np.loadtxt("ds1/ds1_Odometry.dat")
        self.barcodes_data = np.loadtxt("ds1/ds1_Barcodes.dat",unpack=True)

        # Initial state: use first ground truth data
        self.estimated_states = np.array([self.groundtruth_data[0]])
        self.flag=0
        self.measure=measurement_model()
        self.preprocess_data()
        self.generate_particles(num_particles, motion_noise, measurement_noise)
        self.t_prev=self.data[0][0]
        self.particle_filter()
        self.error_x=[]
        self.error_y=[]
        # self.calculate_errors()
    
    def particle_filter(self):
        '''  Implementation of Particle Filter '''
        plt.figure()
        t=[]
        for data in self.data:
            delta_t=data[0]-self.t_prev
            if (data[1] == 0):
                # motion model prediction 
                self.prediction(data,delta_t)
            else:
                # measurement model predcition
                self.update(data)
                self.resampling()
            self.compute_estimate()
            self.t_prev=data[0]
            # Plot every n frames
            if (len(self.estimated_states) > 800 and len(self.estimated_states) % 30 == 0):
                t.append(self.t_prev)
                
                self.plot_filtered_data()
                

        print(len(self.pe))
        print(len(t))
        print(len(self.pg))

   

    def preprocess_data(self):
        ''' preprocess to establish synchronization of data '''

        self.measurement_data=self.clean_data()
        min_time=min(self.measurement_data[0][0],self.odometry_data[0][0])
        odom_data = np.insert(self.odometry_data, 1, 0, axis=1)
        self.data = np.concatenate((odom_data, self.measurement_data), axis = 0)
        self.data = self.data[np.argsort(self.data[:, 0])]
        i=0
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= min_time):
                            break
            i=i+1
        self.groundtruth_data = self.groundtruth_data[i:]
      

    def generate_particles(self, num_particles, motion_noise, measurement_noise):
        ''' Randomly generates a bunch of particles and adds noise to each particle '''
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
    
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.normal(self.estimated_states[-1][1], self.motion_noise[0], num_particles)
        self.particles[:, 1] = np.random.normal(self.estimated_states[-1][2], self.motion_noise[1], num_particles)
        self.particles[:, 2] = np.random.normal(self.estimated_states[-1][3], self.motion_noise[2], num_particles)
        self.last_timestamp = self.estimated_states[-1][0]

        # Initialize wieghts uniformly 
        self.weights = np.full(num_particles, 1.0 / num_particles)


    def mapping(self,barcode_j):
        '''  maps barcode ids to landmark ids '''
        all_ids=list(self.barcodes_data[1])
        barcode_j=int(barcode_j)
        ind=all_ids.index(barcode_j)    
        return self.barcodes_data[0][ind]


    def clean_data(self):
        ''' removes data realted to landmark_id 1-5 ( as the measurement data is coming from other robots ) '''
        robot_ids=[1,2,3,4,5]
        i=0
        x=[]
        
        for i in range(len(self.measurement_data)):
            id=self.measurement_data[i][1]
            landmark_id=self.mapping(id)
            if landmark_id  not in robot_ids:
                x.append(self.measurement_data[i])
            # else:
            #     print(landmark_id,id)
        return np.asarray(x)
    

    def prediction(self,data,delta_t):
            '''
                uses the process model to update the belief in the system state
            '''  
            for particle in self.particles:
                v = np.random.normal(data[2], self.motion_noise[3], 1)
                w = np.random.normal(data[3], self.motion_noise[4], 1)
                if self.flag==0:
                    self.motion=motion_model(particle[0],particle[1],particle[2])
                    self.flag=1

                # particle[0],particle[1] ,particle[2]= self.motion.estimate_state(v,w,delta_t)
                particle[0] += v * np.cos(particle[2]) * delta_t
                particle[1] += v * np.sin(particle[2]) * delta_t
                particle[2] += w * delta_t

                # Limit θ within [-pi, pi]
                if (particle[2] > np.pi):
                    particle[2] -= 2 * np.pi
                elif (particle[2] < -np.pi):
                    particle[2] += 2 * np.pi

    def resampling(self):
        ''' Resamples particles with replacement with probability proportional to their weight '''
        new_idexes = np.random.choice(len(self.particles), len(self.particles), replace = True, p = self.weights)
        # Update new particles according to importance of weights 
        self.particles = self.particles[new_idexes]

    def normal_pdf(self,x, mu, sigma):
        ''' compute its probability density function (PDF)'''
        return 1.0 / (sigma * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))

    def update(self, data):
        '''  Update the state of the particle filter given an observation '''
        landmark_id=self.mapping(data[1])
        x_l,y_l=self.measure.true_landmarks_location(landmark_id)
        for i in range(len(self.particles)):
            # Compute expected range and bearing given current pose
            x_t = self.particles[i][0]
            y_t = self.particles[i][1]
            theta_t = self.particles[i][2]
            range_expected,bearing_expected=self.measure.predicted_landmarks_measurement(x_t,y_t,theta_t)
            # Compute the probability of range and bearing differences in normal distribution with mean = 0 and sigma = measurement noise
            range_error = data[2] - range_expected
            bearing_error = data[3] - bearing_expected
            range_p = self.normal_pdf(0,range_error, self.measurement_noise[0])
            bearing_p =self.normal_pdf(0,bearing_error, self.measurement_noise[1])
            # calc weights
            self.weights[i] = range_p * bearing_p
    
        if (np.sum(self.weights) == 0):
            self.weights = np.ones_like(self.weights)
        self.weights /= np.sum(self.weights)

    def compute_estimate(self):
        ''' computes weighted mean of the set of particles to get a state estimate. '''
        state = np.mean(self.particles, axis = 0)
        self.estimated_states = np.append(self.estimated_states, np.array([[self.last_timestamp, state[0], state[1], state[2]]]), axis = 0)

    def plot_filtered_data(self):
        ''' plotting the data after implementing particle filter '''
        plt.cla()
        plot_e=[self.estimated_states[x][:] for x in range(0,len(self.estimated_states),100)]
        plot_g=[self.groundtruth_data[x][:] for x in range(0,50000,100)]

        # print(len(self.estimated_states))
        # plot_e=[self.estimated_states[x][:] forclen range(0,1000,10)]

        self.pe=plot_e
        self.pg=plot_g
        plot_g=np.asarray(plot_g).transpose()
        plot_e=np.asarray(plot_e).transpose()

        plt.plot(plot_e[1],plot_e[2],'r->',label='robot state estimate')
        plt.plot(plot_g[1],plot_g[2],'b->',label='robot state ground truth')

        

        # # plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b->', label="Robot State Ground truth")
        # # plt.plot(self.estimated_states[:, 1], self.estimated_states[:, 2], 'r->', label="Robot State Estimate")

        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'go', label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'yo', label="End point")
        # Particles
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=5, c='k', alpha=0.5, label="Particles")
        plt.title(" State estimation using particle filter")
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.legend()
        plt.pause(1e-16)

    # def calculate_errors(self):
    #         plt.figure()
    #         p=[0.1,0.5,1,1.5]
    #         # print('error is x',np.mean(self.estimated_states[0:800][1]-self.groundtruth_data[0:800][1]))
    #         # print('error in y',np.mean(self.estimated_states[0:800][2]-self.groundtruth_data[0:800][2]))
    #         # print('errro in theta',np.mean(self.estimated_states[0:800][3]-self.groundtruth_data[0:800][3]))
    #         # error_x=[0.042412611739742284,0.046754845284469465,0.14931592605757535,0.33575346849484367]

    #         # error_y=[0.029699827223990588, 0.020740114697876066,0.09189992347042669,0.12761540281082148]

    #         # error_theta=[0.031529749600937934,0.018078749794874782,0.0935397386719215,-0.15960130582227774]

    #         fig, ax = plt.subplots(3)
    #         plt.suptitle('Error with different noise paramters ')
    #         ax[0].plot(p,error_x,color="blue")
    #         ax[0].set_ylabel('error in x direction(m)')
    #         ax[0].set_xlabel('noise')

    #         ax[1].plot(p,error_y,color="red")
    #         ax[1].set_ylabel('error in y direction(m)')
    #         ax[1].set_xlabel('noise')

    #         ax[2].plot(p,error_theta,color="green")
    #         ax[2].set_ylabel('error in heading(radians)')
    #         ax[2].set_xlabel('noise')
        

                
class plot_data(ParticleFilter):
    def __init__(self):
        self.odom_data=np.loadtxt('ds1/ds1_Odometry.dat',unpack=True)
        self.groundtruth_data=np.loadtxt('ds1/ds1_Groundtruth.dat')
        # initialize motion model with the starting point of the robot 
        # print(self.groundtruth_data[0][1],self.groundtruth_data[0][2],self.groundtruth_data[0][3])
        self.M=motion_model(self.groundtruth_data[0][1],self.groundtruth_data[0][2],self.groundtruth_data[0][3])

    def plot_test_model(self,v_input,w_input,delta_t):
        ''' plots states which are estimated by Motion Model on 5 control inputs(v,w) '''
        state=np.zeros((len(v_input),4))
        total_time=0
        for i in range(len(v_input)):
            # get estimated state 
            state[i][1],state[i][2],state[i][3]=self.M.estimate_state(v_input[i],w_input[i],delta_t[i])
            total_time=total_time+delta_t[i]
            state[i][0]=total_time
            
        fig, ax = plt.subplots(3)
        plt.suptitle('State estimation using Motion Model on 5 control inputs(v,w)')
        ax[0].plot(state[:,0],state[:,1],color="blue")
        ax[0].scatter(state[:,0],state[:,1],color="blue")
        ax[0].set_ylabel('distance in x direction(m)')
        ax[0].set_xlabel('time(s)')

        ax[1].plot(state[:,0],state[:,2],color="red")
        ax[1].scatter(state[:,0],state[:,2],color="red")
        ax[1].set_ylabel('distance in y direction(m)')
        ax[1].set_xlabel('time(s)')

        ax[2].plot(state[:,0],state[:,3],color="green")
        ax[2].scatter(state[:,0],state[:,3],color="green")
        ax[2].set_ylabel('heading(radians)')
        ax[2].set_xlabel('time(s)')

        return state

    def plot_estimtated_state(self):
        ''' plots estimated states on robot dataset with control commands issued by ds1_odometry.dat '''
        _,dataset_len=self.odom_data.shape
        time_prev=0
        self.expected_states=np.zeros((dataset_len,4))
        for i in range(dataset_len):
            # if two odometry data are too close ignore, else estimate new state for the robot
            if self.odom_data[0][i]-time_prev>0.01:
                # calculate delta time 
                delta_time=self.odom_data[0][i]-time_prev
                self.expected_states[i][1],self.expected_states[i][2],self.expected_states[i][3]= self.M.estimate_state(self.odom_data[1][i],self.odom_data[2][i],delta_time)
            time_prev=self.odom_data[0][i]

        # considering starting point as t=0 so subtract all the times to the first time-stamp 
        tstart=self.odom_data[0][0]  # get the first timestamp value
        time=[x-tstart for x in self.odom_data[0,:]]   #adjust all other times accordingly 
        self.expected_states[:,0]=time
        
        fig, ax = plt.subplots(3)
        plt.suptitle('State estimation of robot with control propogation only ')
        ax[0].plot(self.expected_states[:,0],self.expected_states[:,1],color="blue")
        ax[0].set_ylabel('distance in x direction(m)')
        ax[0].set_xlabel('time(s)')

        ax[1].plot(self.expected_states[:,0],self.expected_states[:,2],color="red")
        ax[1].set_ylabel('distance in y direction(m)')
        ax[1].set_xlabel('time(s)')

        ax[2].plot(self.expected_states[:,0],self.expected_states[:,3],color="green")
        ax[2].set_ylabel('heading(radians)')
        ax[2].set_xlabel('time(s)')
    
    def plot_path(self):
        '''  visualizes ground truth path and dead_reckoned path(i.e controls propogation only) '''
        plt.figure()
        plot_e=[self.expected_states[x][:] for x in range(0,len(self.expected_states),50)]
        plot_g=[self.groundtruth_data[x][:] for x in range(0,len(self.groundtruth_data),500)]
        plot_g=np.asarray(plot_g).transpose()
        plot_e=np.asarray(plot_e).transpose()
        plt.plot(plot_e[1],plot_e[2],'g->',label='robot state estimate')
        plt.plot(plot_g[1],plot_g[2],'b->',label='robot state ground truth')       
        plt.plot(plot_e[1][0],plot_e[2][2],'go',label="start point")
        plt.plot(plot_g[1][0],plot_g[2][0],'yo',label="true start point")
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title('Comparision of ground truth path and dead_reckoned path(i.e controls propogation only)')
        plt.legend()
    
def predict_measurement(x,y,theta,j):
    ''' calculates range and bearing to the landmark of id j 

        Input
            x,y,theta:robot's estimated state
            j : landmark's id
    '''
    m=measurement_model()
    # determine true position of landmark with id j
    m.true_landmarks_location(j)
    # predict range values provided landmanrk(x,y) and robot's state (x,y,theta)
    range_mean,bearing_mean=m.predicted_landmarks_measurement(x,y,theta)

    return range_mean,bearing_mean 

def convert_to_global_position(r,b,x,y,theta):
    ''' returns the global position of a landmark  
        Input:
          (r,b) : range and bearing from measurement_model
          (x,y,theta) : esimtated pose of robot from motion_model
        
    '''
    return x+r*np.cos(theta+b),y+r*np.sin(theta+b),theta+b


if __name__ == "__main__":

    p=plot_data()
    # get plot for following control inputs 
    v_input=[0.5,0,0.5,0,0.5]
    w_input=[0,-0.5*np.pi,0,0.5*np.pi,0]
    delta_t=[1,1,1,1,1]

    states=p.plot_test_model(v_input,w_input,delta_t)
    p.plot_estimtated_state()
    p.plot_path()

    print('range and bearing for landmark with id :{} and robot pose x:{},y:{},heading:{} and is {} '.format(6,2,3,0,predict_measurement( 2, 3, 0, 6)))
    print('range and bearing for landmark with id :{} and robot pose x:{},y:{},heading:{} and is {}'.format(13,0,3,0,predict_measurement( 0, 3, 0 ,13)))
    print('range and bearing for landmark with id :{} and robot pose x:{},y:{},heading:{} and is  {}'.format(17,1,-2,0,predict_measurement( 1,-2,0,17)))

    # # #################################  part6 optional ########################
    r,b=predict_measurement(2, 3, 0, 6)
    x_predict ,y_predict,heading =convert_to_global_position(r,b,2,3,0)
    print('converting prediction to global position - error is x:{}, error in y:{}, error in heading:{} '.format(abs(x_predict-2),abs(y_predict-3),abs(heading-0)))

    r,b=predict_measurement(0, 3, 0 ,13)
    x_predict ,y_predict,heading =convert_to_global_position(r,b,0,3,0)
    print('converting prediction to global position - error is x:{}, error in y:{}, error in heading:{} '.format(abs(x_predict-0),abs(y_predict-3),abs(heading-0)))

    r,b=predict_measurement(1,2,0,17)
    x_predict ,y_predict,heading =convert_to_global_position(r,b,1,-2,0)
    print('converting prediction to global position - error is x:{}, error in y:{}, error in heading:{} '.format(abs(x_predict-1),abs(y_predict+2),abs(heading-0)))

    # Particle filter parameters
    num_particles = 100
    motion_noise = np.array([0.1,0.1,0.1, 0.1, 0.1])
    measurement_noise = np.array([0.1,0.1])
    filter = ParticleFilter(num_particles, motion_noise, measurement_noise)
    # plt.show()