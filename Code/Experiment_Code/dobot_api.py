import socket
import numpy as np
import os
import re
import json
import threading
import time
from time import sleep
import requests

alarmControllerFile = "files/alarmController.json"
alarmServoFile = "files/alarmServo.json"

#brief dobot_v4_api:CRA\E6\CRAF\NovaLite
#author futingxing
#date 2025-12-15

# Port Feedback
MyType = np.dtype([('len', np.uint16,),
                   ('reserve', np.byte, (6, )),
                   ('DigitalInputs', np.uint64,),
                   ('DigitalOutputs', np.uint64,),
                   ('RobotMode', np.uint64,),
                   ('TimeStamp', np.uint64,),
                   ('RunTime', np.uint64,),
                   ('TestValue', np.uint64,),
                   ('reserve2', np.byte, (8, )),
                   ('SpeedScaling', np.float64,),
                   ('reserve3', np.byte, (16, )),
                   ('VRobot', np.float64, ),      
                   ('IRobot', np.float64,),
                   ('ProgramState', np.float64,),
                   ('SafetyOIn', np.uint16,),
                   ('SafetyOOut', np.uint16,),
                   ('reserve4', np.byte, (76, )),
                   ('QTarget', np.float64, (6, )),
                   ('QDTarget', np.float64, (6, )),
                   ('QDDTarget', np.float64, (6, )),
                   ('ITarget', np.float64, (6, )),
                   ('MTarget', np.float64, (6, )),
                   ('QActual', np.float64, (6, )),
                   ('QDActual', np.float64, (6, )),
                   ('IActual', np.float64, (6, )),
                   ('ActualTCPForce', np.float64, (6, )),
                   ('ToolVectorActual', np.float64, (6, )),
                   ('TCPSpeedActual', np.float64, (6, )),
                   ('TCPForce', np.float64, (6, )),
                   ('ToolVectorTarget', np.float64, (6, )),
                   ('TCPSpeedTarget', np.float64, (6, )),
                   ('MotorTemperatures', np.float64, (6, )),
                   ('JointModes', np.float64, (6, )),
                   ('VActual', np.float64, (6, )),
                   ('HandType', np.byte, (4, )),
                   ('User', np.byte,),
                   ('Tool', np.byte,),
                   ('RunQueuedCmd', np.byte,),
                   ('PauseCmdFlag', np.byte,),
                   ('VelocityRatio', np.byte,),
                   ('AccelerationRatio', np.byte,),
                   ('reserve5', np.byte, ),
                   ('XYZVelocityRatio', np.byte,),
                   ('RVelocityRatio', np.byte,),
                   ('XYZAccelerationRatio', np.byte,),
                   ('RAccelerationRatio', np.byte,),
                   ('reserve6', np.byte,(2,)),
                   ('BrakeStatus', np.byte,),
                   ('EnableStatus', np.byte,),
                   ('DragStatus', np.byte,),
                   ('RunningStatus', np.byte,),
                   ('ErrorStatus', np.byte,),
                   ('JogStatusCR', np.byte,),   
                   ('CRRobotType', np.byte,),
                   ('DragButtonSignal', np.byte,),
                   ('EnableButtonSignal', np.byte,),
                   ('RecordButtonSignal', np.byte,),
                   ('ReappearButtonSignal', np.byte,),
                   ('JawButtonSignal', np.byte,),
                   ('SixForceOnline', np.byte,),
                   ('CollisionState', np.byte,),
                   ('ArmApproachState', np.byte,),
                   ('J4ApproachState', np.byte,),
                   ('J5ApproachState', np.byte,),
                   ('J6ApproachState', np.byte,),
                   ('reserve7', np.byte, (61, )),
                   ('VibrationDisZ', np.float64,),
                   ('CurrentCommandId', np.uint64,),
                   ('MActual', np.float64, (6, )),
                   ('Load', np.float64,),
                   ('CenterX', np.float64,),
                   ('CenterY', np.float64,),
                   ('CenterZ', np.float64,),
                   ('UserValue[6]', np.float64, (6, )),
                   ('ToolValue[6]', np.float64, (6, )),
                   ('reserve8', np.byte, (8, )),
                   ('SixForceValue', np.float64, (6, )),
                   ('TargetQuaternion', np.float64, (4, )),
                   ('ActualQuaternion', np.float64, (4, )),
                   ('AutoManualMode', np.uint16, ),
                   ('ExportStatus', np.uint16, ),
                   ('SafetyState', np.byte, ),
                   ('reserve9', np.byte,(19,))
                   ])

# Vendor Dobot TCP/IP API documentation note.
# Read controller and servo alarm files
def alarmAlarmJsonFile():
    currrntDirectory = os.path.dirname(__file__)
    jsonContrellorPath = os.path.join(currrntDirectory, alarmControllerFile)
    jsonServoPath = os.path.join(currrntDirectory, alarmServoFile)

    with open(jsonContrellorPath, encoding='utf-8') as f:
        dataController = json.load(f)
    with open(jsonServoPath, encoding='utf-8') as f:
        dataServo = json.load(f)
    return dataController, dataServo

# Vendor Dobot TCP/IP API documentation note.
# TCP communication interface


class DobotApi:
    def __init__(self, ip, port, *args):
        self.ip = ip
        self.port = port
        self.socket_dobot = 0
        self.__globalLock = threading.Lock()
        if args:
            self.text_log = args[0]

        if self.port == 29999 or self.port == 30004 or self.port == 30005:
            try:
                self.socket_dobot = socket.socket()
                self.socket_dobot.connect((self.ip, self.port))
                self.socket_dobot.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 144000)
            except socket.error:
                print(socket.error)

        else:
            print(f"Connect to dashboard server need use port {self.port} !")

    def log(self, text):
        if self.text_log:
            print(text)

    def send_data(self, string):
       # self.log(f"Send to {self.ip}:{self.port}: {string}")
        try:
            self.socket_dobot.send(str.encode(string, 'utf-8'))
        except Exception as e:
            print(e)
            while True:
                try:
                    self.socket_dobot = self.reConnect(self.ip, self.port)
                    self.socket_dobot.send(str.encode(string, 'utf-8'))
                    break
                except Exception:
                    sleep(1)

    def wait_reply(self):
        """
        Read the return value
        """
        data = ""
        try:
            data = self.socket_dobot.recv(1024)
        except Exception as e:
            print(e)
            self.socket_dobot = self.reConnect(self.ip, self.port)

        finally:
            if len(data) == 0:
                data_str = data
            else:
                data_str = str(data, encoding="utf-8")
            # self.log(f'Receive from {self.ip}:{self.port}: {data_str}')
            return data_str

    def close(self):
        """
        Close the port
        """
        if (self.socket_dobot != 0):
            try:
                self.socket_dobot.shutdown(socket.SHUT_RDWR)
                self.socket_dobot.close()
            except socket.error as e:
                print(f"Error while closing socket: {e}")

    def sendRecvMsg(self, string):
        """
        send-recv Sync
        """
        with self.__globalLock:
            self.send_data(string)
            recvData = self.wait_reply()
            return recvData

    def __del__(self):
        self.close()

    def reConnect(self, ip, port):
        while True:
            try:
                socket_dobot = socket.socket()
                socket_dobot.connect((ip, port))
                break
            except Exception:
                sleep(1)
        return socket_dobot

# Vendor Dobot TCP/IP API documentation note.
# Control and motion command interface


class DobotApiDashboard(DobotApi):

    def __init__(self, ip, port, *args):
        super().__init__(ip, port, *args)

    def _fmt(self, v):
        if isinstance(v, (list, tuple)):
            return "{" + ",".join([self._fmt(x) for x in v]) + "}"
        if isinstance(v, float):
            return ("{:f}".format(v))
        if isinstance(v, int):
            return ("{:d}".format(v))
        return str(v)

    def _build_cmd(self, name, *args, **kwargs):
        parts = []
        for a in args:
            parts.append(self._fmt(a))
        for k, v in kwargs.items():
            parts.append(f"{k}={self._fmt(v)}")
        return f"{name}(" + ",".join(parts) + ")"

    def EnableRobot(self, load=0.0, centerX=0.0, centerY=0.0, centerZ=0.0, isCheck=-1,):
        'Vendor Dobot TCP/IP API wrapper method.'
        """
            Optional parameter
            Parameter name     Type     Description
            load     double     Load weight. The value range should not exceed the load range of corresponding robot models. Unit: kg.
            centerX     double     X-direction eccentric distance. Range: -999 – 999, unit: mm.
            centerY     double     Y-direction eccentric distance. Range: -999 – 999, unit: mm.
            centerZ     double     Z-direction eccentric distance. Range: -999 – 999, unit: mm.
            isCheck     int     Check the load or not. 1: check, 0: not check. If set to 1, the robot arm will check whether the actual load is the same as the set load after it is enabled, and if not, it will be automatically disabled. 0 by default.
            The number of parameters that can be contained is as follows:
            0: no parameter (not set load weight and eccentric parameters when enabling the robot).
            1: one parameter (load weight).
            4: four parameters (load weight and eccentric parameters).
            5: five parameters (load weight, eccentric parameters, check the load or not).
                """
        string = 'EnableRobot('
        if load != 0:
            string = string + "{:f}".format(load)
            if centerX != 0 or centerY != 0 or centerZ != 0:
                string = string + ",{:f},{:f},{:f}".format(
                    centerX, centerY, centerZ)
                if isCheck != -1:
                    string = string + ",{:d}".format(isCheck)
        string = string + ')'
        return self.sendRecvMsg(string)

    def DisableRobot(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DisableRobot()"
        return self.sendRecvMsg(string)

    def ClearError(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ClearError()"
        return self.sendRecvMsg(string)

    def PowerOn(self):
        """
        Powering on the robot
        Note: It takes about 10 seconds for the robot to be enabled after it is powered on.
        """
        string = "PowerOn()"
        return self.sendRecvMsg(string)

    def RunScript(self, project_name):
        """
        Run the script file
        project_name ：Script file name
        """
        string = "RunScript({:s})".format(project_name)
        return self.sendRecvMsg(string)

    def Stop(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "Stop()"
        return self.sendRecvMsg(string)

    def Pause(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "Pause()"
        return self.sendRecvMsg(string)

    def Continue(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "Continue()"
        return self.sendRecvMsg(string)

    def EmergencyStop(self, mode):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "EmergencyStop({:d})".format(mode)
        return self.sendRecvMsg(string)

    def BrakeControl(self, axisID, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "BrakeControl({:d},{:d})".format(axisID, value)
        return self.sendRecvMsg(string)

    #####################################################################

    def SpeedFactor(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SpeedFactor({:d})".format(speed)
        return self.sendRecvMsg(string)

    def User(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "User({:d})".format(index)
        return self.sendRecvMsg(string)

    def SetUser(self, index, table):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetUser({:d},{:s})".format(index, table)
        return self.sendRecvMsg(string)

    def CalcUser(self, index, matrix_direction, table):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "CalcUser({:d},{:d},{:s})".format(
            index, matrix_direction, table)
        return self.sendRecvMsg(string)

    def Tool(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "Tool({:d})".format(index)
        return self.sendRecvMsg(string)

    def SetTool(self, index, table):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetTool({:d},{:s})".format(index, table)
        return self.sendRecvMsg(string)

    def CalcTool(self, index, matrix_direction, table):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "CalcTool({:d},{:d},{:s})".format(
            index, matrix_direction, table)
        return self.sendRecvMsg(string)

    def SetPayload(self, load=0.0, X=0.0, Y=0.0, Z=0.0, name='F'):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = 'SetPayload('
        if name != 'F':
            string = string + "{:s}".format(name)
        else:
            if load != 0:
                string = string + "{:f}".format(load)
                if X != 0 or Y != 0 or Z != 0:
                    string = string + ",{:f},{:f},{:f}".format(X, Y, Z)
        string = string + ')'
        return self.sendRecvMsg(string)

    def AccJ(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "AccJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def AccL(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "AccL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def VelJ(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "VelJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def VelL(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "VelL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def CP(self, ratio):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "CP({:d})".format(ratio)
        return self.sendRecvMsg(string)

    def SetCollisionLevel(self, level):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetCollisionLevel({:d})".format(level)
        return self.sendRecvMsg(string)

    def SetBackDistance(self, distance):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetBackDistance({:d})".format(distance)
        return self.sendRecvMsg(string)

    def SetPostCollisionMode(self, mode):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetPostCollisionMode({:d})".format(mode)
        return self.sendRecvMsg(string)

    def StartDrag(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "StartDrag()"
        return self.sendRecvMsg(string)

    def StopDrag(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "StopDrag()"
        return self.sendRecvMsg(string)

    def DragSensivity(self, index, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DragSensivity({:d},{:d})".format(index, value)
        return self.sendRecvMsg(string)

    def EnableSafeSkin(self, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "EnableSafeSkin({:d})".format(status)
        return self.sendRecvMsg(string)

    def SetSafeSkin(self, part, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetSafeSkin({:d},{:d})".format(part, status)
        return self.sendRecvMsg(string)

    def SetSafeWallEnable(self, index, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetSafeWallEnable({:d},{:d})".format(index, value)
        return self.sendRecvMsg(string)

    def SetWorkZoneEnable(self, index, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetWorkZoneEnable({:d},{:d})".format(index, value)
        return self.sendRecvMsg(string)

    #########################################################################

    def RobotMode(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RobotMode()"
        return self.sendRecvMsg(string)

    def PositiveKin(self, J1, J2, J3, J4, J5, J6, user=-1, tool=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "PositiveKin({:f},{:f},{:f},{:f},{:f},{:f}".format(
            J1, J2, J3, J4, J5, J6)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def InverseKin(self, X, Y, Z, Rx, Ry, Rz, user=-1, tool=-1, useJointNear=-1, JointNear=''):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "InverseKin({:f},{:f},{:f},{:f},{:f},{:f}".format(
            X, Y, Z, Rx, Ry, Rz)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if useJointNear != -1:
            params.append('useJointNear={:d}'.format(useJointNear))
        if JointNear != '':
            params.append('JointNear={:s}'.format(JointNear))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def GetAngle(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetAngle()"
        return self.sendRecvMsg(string)

    def GetPose(self, user=-1, tool=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetPose("
        params = []
        state = True
        if user != -1:
            params.append('user={:d}'.format(user))
            state = not state
        if tool != -1:
            params.append('tool={:d}'.format(tool))
            state = not state
        if not state:
            return 'need to be set or not set at the same time. They are global user coordinate system and global tool coordinate system if not set' # Vendor Dobot TCP/IP API documentation note.

        for i, param in enumerate(params):
            if i == len(params)-1:
                string = string + param
            else:
                string = string + param+","

        string = string + ')'
        return self.sendRecvMsg(string)

    def GetErrorID(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetErrorID()"
        return self.sendRecvMsg(string)

     #################################################################

    def DO(self, index, status, time=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DO({:d},{:d}".format(index, status)
        params = []
        if time != -1:
            params.append('{:d}'.format(time))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def DOInstant(self, index, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DOInstant({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def GetDO(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetDO({:d})".format(index)
        return self.sendRecvMsg(string)

    def DOGroup(self, *index_value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DOGroup({:d}".format(index_value[0])
        for ii in index_value[1:]:
            string = string + ',' + str(ii)
        string = string + ')'

        return self.sendRecvMsg(string)

    def GetDOGroup(self, *index_value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetDOGroup({:d}".format(index_value[0])
        for ii in index_value[1:]:
            string = string + ',' + str(ii)
        string = string + ')'
        return self.sendRecvMsg(string)

    def ToolDO(self, index, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ToolDO({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def ToolDOInstant(self, index, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ToolDOInstant({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def GetToolDO(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetToolDO({:d})".format(index)
        return self.sendRecvMsg(string)

    def AO(self, index, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "AO({:d},{:f})".format(index, value)
        return self.sendRecvMsg(string)

    def AOInstant(self, index, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "AOInstant({:d},{:f})".format(index, value)
        return self.sendRecvMsg(string)

    def GetAO(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetAO({:d})".format(index)
        return self.sendRecvMsg(string)

    def DI(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DI({:d})".format(index)
        return self.sendRecvMsg(string)

    def DIGroup(self, *index_value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "DIGroup({:d}".format(index_value[0])
        for ii in index_value[1:]:
            string = string + ',' + str(ii)
        string = string + ')'
        return self.sendRecvMsg(string)

    def ToolDI(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ToolDI({:d})".format(index)
        return self.sendRecvMsg(string)

    def AI(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "AI({:d})".format(index)
        return self.sendRecvMsg(string)

    def ToolAI(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ToolAI({:d})".format(index)
        return self.sendRecvMsg(string)

    def SetTool485(self, index, parity='', stopbit=-1, identify=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetTool485({:d}".format(index)
        params = []
        if parity != '':
            params.append(parity)
        if string != -1:
            params.append('{:d}'.format(stopbit))
            if identify != -1:
                params.append('{:d}'.format(identify))
        else:
            if identify != -1:
                params.append('1,{:d}'.format(identify))  # Vendor Dobot TCP/IP API documentation note.
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def SetToolPower(self, status, identify=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetToolPower({:d}".format(status)
        params = []
        if identify != -1:
            params.append('{:d}'.format(identify))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def SetToolMode(self, mode, type, identify=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetToolMode({:d},{:d}".format(mode, type)
        params = []
        if identify != -1:
            params.append('{:d}'.format(identify))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

     ##################################################################

    def ModbusCreate(self, ip, port, slave_id, isRTU=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ModbusCreate({:s},{:d},{:d}".format(ip, port, slave_id)
        params = []
        if isRTU != -1:
            params.append('{:d}'.format(isRTU))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def ModbusRTUCreate(self, slave_id, baud, parity='', data_bit=8, stop_bit=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ModbusRTUCreate({:d},{:d}".format(slave_id, baud)
        params = []
        if parity != '':
            params.append('{:s}'.format(parity))
        if data_bit != 8:
            params.append('{:d}'.format(data_bit))
        if stop_bit != -1:
            params.append('{:d}'.format(stop_bit))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def ModbusClose(self, index):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ModbusClose({:d})".format(index)
        return self.sendRecvMsg(string)

    def GetInBits(self, index, addr, count):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInBits({:d},{:d},{:d})".format(index, addr, count)
        return self.sendRecvMsg(string)

    def GetInRegs(self, index, addr, count, valType=''):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInRegs({:d},{:d},{:d}".format(index, addr, count)
        params = []
        if valType != '':
            params.append('{:s}'.format(valType))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def GetCoils(self, index, addr, count):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetCoils({:d},{:d},{:d})".format(index, addr, count)
        return self.sendRecvMsg(string)

    def SetCoils(self, index, addr, count, valTab):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetCoils({:d},{:d},{:d},{:s})".format(
            index, addr, count, valTab)
        return self.sendRecvMsg(string)

    def GetHoldRegs(self, index, addr, count, valType=''):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetHoldRegs({:d},{:d},{:d}".format(index, addr, count)
        params = []
        if valType != '':
            params.append('{:s}'.format(valType))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def SetHoldRegs(self, index, addr, count, valTab, valType=''):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetHoldRegs({:d},{:d},{:d},{:s}".format(
            index, addr, count, valTab)
        params = []
        if valType != '':
            params.append('{:s}'.format(valType))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)
    ########################################################################

    def GetInputBool(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInputBool({:d})".format(address)
        return self.sendRecvMsg(string)

    def GetInputInt(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInputInt({:d})".format(address)
        return self.sendRecvMsg(string)

    def GetInputFloat(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInputFloat({:d})".format(address)
        return self.sendRecvMsg(string)

    def GetOutputBool(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetOutputBool({:d})".format(address)
        return self.sendRecvMsg(string)

    def GetOutputInt(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetOutputInt({:d})".format(address)
        return self.sendRecvMsg(string)

    def GetOutputFloat(self, address):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInputFloat({:d})".format(address)
        return self.sendRecvMsg(string)

    def SetOutputBool(self, address, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetInputFloat({:d},{:d})".format(address, value)
        return self.sendRecvMsg(string)

    def SetOutputInt(self, address, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetOutputInt({:d},{:d})".format(address, value)
        return self.sendRecvMsg(string)

    def SetOutputFloat(self, address, value):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetOutputFloat({:d},{:d})".format(address, value)
        return self.sendRecvMsg(string)

    #######################################################################

    def MovJ(self, a1, b1, c1, d1, e1, f1, coordinateMode, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "MovJ(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1)
        elif coordinateMode == 1:
            string = "MovJ(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1)
        else:
            print("coordinateMode param is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def MovL(self, a1, b1, c1, d1, e1, f1, coordinateMode, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "MovL(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1)
        elif coordinateMode == 1:
            string = "MovL(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def ServoJ(self, J1, J2, J3, J4, J5, J6, t=-1.0,aheadtime=-1.0, gain=-1.0):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "ServoJ({:f},{:f},{:f},{:f},{:f},{:f}".format(J1, J2, J3, J4, J5, J6)
        params = []
        if t != -1:
            params.append('t={:f}'.format(t))
        if aheadtime != -1:
            params.append('aheadtime={:f}'.format(aheadtime))
        if gain != -1:
            params.append('gain={:f}'.format(gain))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)
    def ServoP(self, X, Y, Z, RX, RY, RZ, t=-1.0,aheadtime=-1.0, gain=-1.0):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "ServoP({:f},{:f},{:f},{:f},{:f},{:f}".format(X, Y, Z, RX, RY, RZ)
        params = []
        if t != -1:
            params.append('t={:f}'.format(t))
        if aheadtime != -1:
            params.append('aheadtime={:f}'.format(aheadtime))
        if gain != -1:
            params.append('gain={:f}'.format(gain))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def MovLIO(self, a1, b1, c1, d1, e1, f1, coordinateMode, Mode, Distance, Index, Status, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "MovLIO(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:d},{:d},{:d},{:d}}}".format(
                a1, b1, c1, d1, e1, f1, Mode, Distance, Index, Status)
        elif coordinateMode == 1:
            string = "MovLIO(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:d},{:d},{:d},{:d}}}".format(
                a1, b1, c1, d1, e1, f1, Mode, Distance, Index, Status)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def MovJIO(self,  a1, b1, c1, d1, e1, f1, coordinateMode, Mode, Distance, Index, Status, user=-1, tool=-1, a=-1, v=-1, cp=-1,):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "MovJIO(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:d},{:d},{:d},{:d}}}".format(
                a1, b1, c1, d1, e1, f1, Mode, Distance, Index, Status)
        elif coordinateMode == 1:
            string = "MovJIO(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:d},{:d},{:d},{:d}}}".format(
                a1, b1, c1, d1, e1, f1, Mode, Distance, Index, Status)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def Arc(self, a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2, coordinateMode, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "Arc(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2)
        elif coordinateMode == 1:
            string = "Arc(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def Circle(self, a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2, coordinateMode, count, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "Circle(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},{:d}".format(
                a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2, count)
        elif coordinateMode == 1:
            string = "Circle(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},{:d}".format(
                a1, b1, c1, d1, e1, f1,  a2, b2, c2, d2, e2, f2, count)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def MoveJog(self, axis_id='', coordtype=-1, user=-1, tool=-1):
        """
        Joint motion
        axis_id: Joint motion axis, optional string value:
            J1+ J2+ J3+ J4+ J5+ J6+
            J1- J2- J3- J4- J5- J6- 
            X+ Y+ Z+ Rx+ Ry+ Rz+ 
            X- Y- Z- Rx- Ry- Rz-
        *dynParams: Parameter Settings（coord_type, user_index, tool_index）
                    coord_type: 1: User coordinate 2: tool coordinate (default value is 1)
                    user_index: user index is 0 ~ 9 (default value is 0)
                    tool_index: tool index is 0 ~ 9 (default value is 0)
        """
        string = "MoveJog({:s}".format(axis_id)
        params = []
        if coordtype != -1:
            params.append('coordtype={:d}'.format(coordtype))
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def GetStartPose(self, trace_name):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetStartPose({:s})".format(trace_name)
        return self.sendRecvMsg(string)

    def StartPath(self, trace_name, isConst=-1, multi=-1.0, user=-1, tool=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "StartPath({:s}".format(trace_name)
        params = []
        if isConst != -1:
            params.append('isConst={:d}'.format(isConst))
        if multi != -1:
            params.append('multi={:f}'.format(multi))
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelMovJTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RelMovJTool({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelMovLTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RelMovLTool({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelMovJUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RelMovJUser({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelMovLUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RelMovLUser({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelJointMovJ(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, a=-1, v=-1, cp=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "RelJointMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        params = []
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1:
            params.append('cp={:d}'.format(cp))
        for ii in params:
            string = string + ',' + ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def GetCurrentCommandID(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetCurrentCommandID()"
        return self.sendRecvMsg(string)

    # Vendor Dobot TCP/IP API documentation note.
    
    # Vendor Dobot TCP/IP API documentation note.
    def SetResumeOffset(self, distance):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SetResumeOffset({:f})".format(distance)
        return self.sendRecvMsg(string)
    
    def PathRecovery(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "PathRecovery()"
        return self.sendRecvMsg(string)

    def PathRecoveryStop(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "PathRecoveryStop()"
        return self.sendRecvMsg(string)

    def PathRecoveryStatus(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "PathRecoveryStatus()"
        return self.sendRecvMsg(string)
    
    # Vendor Dobot TCP/IP API documentation note.
    def LogExportUSB(self, range):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "LogExportUSB({:d})".format(range)
        return self.sendRecvMsg(string)
    
    def GetExportStatus(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "GetExportStatus()"
        return self.sendRecvMsg(string)

    # Vendor Dobot TCP/IP API documentation note.
    def EnableFTSensor(self, status):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "EnableFTSensor({:d})".format(status)
        return self.sendRecvMsg(string)

    def SixForceHome(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "SixForceHome()"
        return self.sendRecvMsg(string)
    
    def GetForce(self, tool = -1):
        'Vendor Dobot TCP/IP API wrapper method.'
        if tool == -1:
            string = "GetForce()"
        else:
            string = "GetForce({:d})".format(tool)
        return self.sendRecvMsg(string)

    def ForceDriveMode(self, x, y, z, rx, ry, rz, user=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "ForceDriveMode("+"{"+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)+"}"
        if user != -1:
            string = string + ',{:d}'.format(user)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def ForceDriveSpeed(self, speed):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "ForceDriveSpeed({:d})".format(speed)
        return self.sendRecvMsg(string)
    
    def FCForceMode(self, x, y, z, rx, ry, rz, fx,fy,fz,frx,fry,frz, reference=-1, user=-1,tool=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCForceMode("+"{"+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)+"},"+"{"+"{:d},{:d},{:d},{:d},{:d},{:d}".format(fx,fy,fz,frx,fry,frz)+"}"
        params = []
        if reference != -1:
            params.append('reference={:d}'.format(reference))
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        for ii in params:
            string = string + ','+ii
        string = string + ')'
        return self.sendRecvMsg(string)

    def FCSetDeviation(self, x, y, z, rx, ry, rz, controltype=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetDeviation("+"{"+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)+"}"
        if controltype != -1:
            string = string + ',{:d}'.format(controltype)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def FCSetForceLimit(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetForceLimit("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def FCSetMass(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetMass("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def FCSetStiffness(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetStiffness("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def FCSetDamping(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetDamping("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)

    def FCOff(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = "FCOff()"
        return self.sendRecvMsg(string)

    def FCSetForceSpeedLimit(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetForceSpeedLimit("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def FCSetForce(self, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = "FCSetForce("+"{:d},{:d},{:d},{:d},{:d},{:d}".format(x,y,z,rx,ry,rz)
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def RequestControl(self):
        """
        Request control of the robot.
        Note: This function sends a request for the control of the robot, which may be approved or denied.
        """
        string = "RequestControl()"
        return self.sendRecvMsg(string)
    
    # Vendor Dobot TCP/IP API documentation note.

    def RelPointTool(self, coordinateMode,a1, b1, c1, d1, e1, f1, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "RelPointTool(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},".format(
                a1, b1, c1, d1, e1, f1)
        elif coordinateMode == 1:
            string = "RelPointTool(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},".format(
                a1, b1, c1, d1, e1, f1)
        string = string + "{"+"{:f},{:f},{:f},{:f},{:f},{:f}".format(x,y,z,rx,ry,rz)+"}"
        string = string + ')'
        return self.sendRecvMsg(string)
    
    def RelPointUser(self,coordinateMode,a1, b1, c1, d1, e1, f1, x, y, z, rx, ry, rz):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        string = ""
        if coordinateMode == 0:
            string = "RelPointUser(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},".format(
                a1, b1, c1, d1, e1, f1)
        elif coordinateMode == 1:
            string = "RelPointUser(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},".format(
                a1, b1, c1, d1, e1, f1)
        string =string + "{"+"{:f},{:f},{:f},{:f},{:f},{:f}".format(x,y,z,rx,ry,rz)+"}"
        string = string + ')'
        return self.sendRecvMsg(string)

    def RelJoint(self, j1, j2, j3, j4, j5, j6, offset1, offset2, offset3, offset4, offset5, offset6):
        """
        RelJoint command
        """
        string = "RelJoint({:f},{:f},{:f},{:f},{:f},{:f},{{{:f},{:f},{:f},{:f},{:f},{:f}}})".format(
            j1, j2, j3, j4, j5, j6, offset1, offset2, offset3, offset4, offset5, offset6)
        return self.sendRecvMsg(string)
    
    def GetError(self, language="zh_cn"):
        'Vendor Dobot TCP/IP API wrapper method.'
        try:
            # Vendor Dobot TCP/IP API documentation note.
            language_url = f"http://{self.ip}:22000/interface/language"
            language_data = {"type": language}
            
            # Vendor Dobot TCP/IP API documentation note.
            response = requests.post(language_url, json=language_data, timeout=5)
            if response.status_code != 200:
                print(f"Failed to set language: HTTP {response.status_code}")
            
            # Vendor Dobot TCP/IP API documentation note.
            alarm_url = f"http://{self.ip}:22000/protocol/getAlarm"
            response = requests.get(alarm_url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to obtain alarm information: HTTP {response.status_code}")
                return {"errMsg": []}
                
        except requests.exceptions.RequestException as e:
            print(f"HTTP request exception: {e}")
            return {"errMsg": []}
        except json.JSONDecodeError as e:
            print(f"JSON parsing exception: {e}")
            return {"errMsg": []}
        except Exception as e:
            print(f"Unknown error while obtaining alarm information: {e}")
            return {"errMsg": []}

    def ArcIO(self, a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, coordinateMode, *io_params, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1, mode=-1):
        'Vendor Dobot TCP/IP API wrapper method.'
        string = ""
        if coordinateMode == 0:
            string = "ArcIO(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2)
        elif coordinateMode == 1:
            string = "ArcIO(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
                a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2)
        else:
            print("coordinateMode  param  is wrong")
            return ""
        
        for io_param in io_params:
            if isinstance(io_param, (list, tuple)) and len(io_param) == 4:
                string += ",{{{:d},{:d},{:d},{:d}}}".format(*io_param)
            else:
                 print("io_param format is wrong")

        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
        if mode != -1:
            params.append('mode={:d}'.format(mode))
        
        for ii in params:
            string += ',' + ii
        string += ')'
        return self.sendRecvMsg(string)

    def ArcTrackStart(self):
        return self.sendRecvMsg("ArcTrackStart()")

    def ArcTrackParams(self, sampleTime, coordinateType, upDownCompensationMin, upDownCompensationMax, upDownCompensationOffset, leftRightCompensationMin, leftRightCompensationMax, leftRightCompensationOffset):
        string = "ArcTrackParams({:d},{:d},{:f},{:f},{:f},{:f},{:f},{:f})".format(
            sampleTime, coordinateType, upDownCompensationMin, upDownCompensationMax, upDownCompensationOffset, leftRightCompensationMin, leftRightCompensationMax, leftRightCompensationOffset)
        return self.sendRecvMsg(string)

    def ArcTrackEnd(self):
        return self.sendRecvMsg("ArcTrackEnd()")

    def CheckMovC(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, j1c, j2c, j3c, j4c, j5c, j6c, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        string = "CheckMovC(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, j1c, j2c, j3c, j4c, j5c, j6c)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CheckMovJ(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        string = "CheckMovJ(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CheckOddMovC(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, j1c, j2c, j3c, j4c, j5c, j6c, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        string = "CheckOddMovC(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, j1c, j2c, j3c, j4c, j5c, j6c)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CheckOddMovJ(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        string = "CheckOddMovJ(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CheckOddMovL(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, user=-1, tool=-1, a=-1, v=-1, cp=-1):
        string = "CheckOddMovL(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CnvInit(self, index):
        """
        CnvInit command
        """
        string = "CnvInit({:d})".format(index)
        return self.sendRecvMsg(string)

    def CnvMovL(self, j1, j2, j3, j4, j5, j6, user=-1, tool=-1, a=-1, v=-1, cp=-1, r=-1):
        string = "CnvMovL(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1, j2, j3, j4, j5, j6)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if r != -1: params.append('r={:d}'.format(r))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CnvMovC(self, j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b, user=-1, tool=-1, a=-1, v=-1, cp=-1, r=-1, mode=1):
        string = "CnvMovC(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            j1a, j2a, j3a, j4a, j5a, j6a, j1b, j2b, j3b, j4b, j5b, j6b)
        params = []
        if user != -1: params.append('user={:d}'.format(user))
        if tool != -1: params.append('tool={:d}'.format(tool))
        if a != -1: params.append('a={:d}'.format(a))
        if v != -1: params.append('v={:d}'.format(v))
        if cp != -1: params.append('cp={:d}'.format(cp))
        if r != -1: params.append('r={:d}'.format(r))
        if mode != 1: params.append('mode={:d}'.format(mode))
        if params: string += "," + ",".join(params)
        string += ")"
        return self.sendRecvMsg(string)

    def CreateTray(self, *args, **kwargs):
        """
        CreateTray command
        Due to missing documentation on exact parameters, this function uses dynamic arguments.
        Example: CreateTray(rows=3, cols=4, ...)
        """
        return self.sendRecvMsg(self._build_cmd("CreateTray", *args, **kwargs))

    def EndRTOffset(self):
        return self.sendRecvMsg("EndRTOffset()")

    def StartRTOffset(self):
        """
        StartRTOffset command
        """
        return self.sendRecvMsg("StartRTOffset()")

    def FCCollisionSwitch(self, enable):
        return self.sendRecvMsg("FCCollisionSwitch(enable={:d})".format(enable))

    def SetFCCollision(self, force, torque):
        return self.sendRecvMsg("SetFCCollision({:f},{:f})".format(force, torque))

    def GetCnvObject(self, objId):
        return self.sendRecvMsg("GetCnvObject({:d})".format(objId))

    def DOGroupDEC(self, group, value):
        return self.sendRecvMsg("DOGroupDEC({:d},{:d})".format(group, value))

    def GetDOGroupDEC(self, group, value):
        return self.sendRecvMsg("GetDOGroupDEC({:d},{:d})".format(group, value))

    def DIGroupDEC(self, group, value):
        return self.sendRecvMsg("DIGroupDEC({:d},{:d})".format(group, value))

    def InverseSolution(self, a1, b1, c1, d1, e1, f1, user=-1, tool=-1, isJoint=0):
        """
        InverseSolution command
        """
        string = "InverseSolution(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            a1, b1, c1, d1, e1, f1)
        
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if isJoint != 0:
            params.append('isJoint={:d}'.format(isJoint))
            
        for ii in params:
            string += ',' + ii
        string += ')'
        return self.sendRecvMsg(string)

    def MoveL(self, a1, b1, c1, d1, e1, f1, user=-1, tool=-1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        """
        MoveL command
        """
        string = "MoveL(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(
            a1, b1, c1, d1, e1, f1)
        
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if cp != -1 and r != -1:
            params.append('r={:d}'.format(r))
        elif r != -1:
            params.append('r={:d}'.format(r))
        elif cp != -1:
            params.append('cp={:d}'.format(cp))
            
        for ii in params:
            string += ',' + ii
        string += ')'
        return self.sendRecvMsg(string)

    def MovS(self, file=None, coordinateMode=-1, points=None, user=-1, tool=-1, v=-1, speed=-1, a=-1, freq=-1):
        """
        MovS command
        """
        string = "MovS("
        if file is not None:
             string += "file={:s}".format(file)
        elif points is not None and coordinateMode != -1:
             # points should be a list of tuples/lists
             pts_str = []
             for pt in points:
                 if coordinateMode == 0:
                     pts_str.append("pose={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(*pt))
                 elif coordinateMode == 1:
                     pts_str.append("joint={{{:f},{:f},{:f},{:f},{:f},{:f}}}".format(*pt))
             string += ",".join(pts_str)
        else:
             print("MovS param is wrong")
             return ""
        
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if v != -1 and speed != -1:
            params.append('speed={:d}'.format(speed))
        elif speed != -1:
            params.append('speed={:d}'.format(speed))
        elif v != -1:
            params.append('v={:d}'.format(v))
        if a != -1:
             params.append('a={:d}'.format(a))
        if freq != -1:
             params.append('freq={:d}'.format(freq))
             
        if len(params) > 0:
             if file is not None or (points is not None and len(points) > 0):
                  string += ","
             string += ",".join(params)
             
        string += ")"
        return self.sendRecvMsg(string)

    def OffsetPara(self, x, y, z, rx, ry, rz):
        """
        OffsetPara command
        """
        string = "OffsetPara({:f},{:f},{:f},{:f},{:f},{:f})".format(x, y, z, rx, ry, rz)
        return self.sendRecvMsg(string)


    def GetTrayPoint(self, *args, **kwargs):
        """
        GetTrayPoint command
        Due to missing documentation on exact parameters, this function uses dynamic arguments.
        Example: GetTrayPoint(trayName)
        """
        return self.sendRecvMsg(self._build_cmd("GetTrayPoint", *args, **kwargs))

    def ResetRobot(self):
        return self.sendRecvMsg("ResetRobot()")

    def RunTo(self, a1, b1, c1, d1, e1, f1, moveType, user=-1, tool=-1, a=-1, v=-1):
        """
        RunTo command
        """
        string = ""
        if moveType == 0:
            string = "RunTo(pose={{{:f},{:f},{:f},{:f},{:f},{:f}}},moveType=0".format(
                a1, b1, c1, d1, e1, f1)
        elif moveType == 1:
            string = "RunTo(joint={{{:f},{:f},{:f},{:f},{:f},{:f}}},moveType=1".format(
                a1, b1, c1, d1, e1, f1)
        else:
             print("moveType param is wrong")
             return ""
        
        params = []
        if user != -1:
            params.append('user={:d}'.format(user))
        if tool != -1:
            params.append('tool={:d}'.format(tool))
        if a != -1:
            params.append('a={:d}'.format(a))
        if v != -1:
            params.append('v={:d}'.format(v))
        
        for ii in params:
            string += ',' + ii
        string += ')'
        return self.sendRecvMsg(string)

    def SetArcTrackOffset(self, offsetX, offsetY, offsetZ, offsetRx, offsetRy, offsetRz):
        string = "SetArcTrackOffset({{{:f},{:f},{:f},{:f},{:f},{:f}}})".format(
            offsetX, offsetY, offsetZ, offsetRx, offsetRy, offsetRz)
        return self.sendRecvMsg(string)

    def SetCnvPointOffset(self, xOffset, yOffset):
        return self.sendRecvMsg("SetCnvPointOffset({:f},{:f})".format(xOffset, yOffset))

    def SetCnvTimeCompensation(self, time):
        return self.sendRecvMsg("SetCnvTimeCompensation({:d})".format(time))

    def StartSyncCnv(self):
        return self.sendRecvMsg("StartSyncCnv()")

    def StopSyncCnv(self):
        return self.sendRecvMsg("StopSyncCnv()")

    def TcpSendAndParse(self, cmd):
        """
        TcpSendAndParse command
        """
        return self.sendRecvMsg("TcpSendAndParse(\"{:s}\")".format(cmd))

    def Sleep(self, count):
        return self.sendRecvMsg("Sleep({:d})".format(count))

    def RelPointWeldLine(self, StartX, EndX, Y, Z, WorkAngle, TravelAngle, P1, P2):
        string = "RelPointWeldLine({:f},{:f},{:f},{:f},{:f},{:f},{{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:f},{:f},{:f},{:f},{:f},{:f}}})".format(
            StartX, EndX, Y, Z, WorkAngle, TravelAngle, P1[0], P1[1], P1[2], P1[3], P1[4], P1[5], P2[0], P2[1], P2[2], P2[3], P2[4], P2[5])
        return self.sendRecvMsg(string)

    def RelPointWeldArc(self, StartX, EndX, Y, Z, WorkAngle, TravelAngle, P1, P2, P3):
        string = "RelPointWeldArc({:f},{:f},{:f},{:f},{:f},{:f},{{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:f},{:f},{:f},{:f},{:f},{:f}}},{{{:f},{:f},{:f},{:f},{:f},{:f}}})".format(
            StartX, EndX, Y, Z, WorkAngle, TravelAngle, P1[0], P1[1], P1[2], P1[3], P1[4], P1[5], P2[0], P2[1], P2[2], P2[3], P2[4], P2[5], P3[0], P3[1], P3[2], P3[3], P3[4], P3[5])
        return self.sendRecvMsg(string)

    def WeaveStart(self):
        return self.sendRecvMsg("WeaveStart()")

    def WeaveParams(self, weldType, frequency, leftAmplitude, rightAmplitude, direction, stopMode, stopTime1, stopTime2, stopTime3, stopTime4, radius, radian, **kwargs):
        string = "WeaveParams({:d},{:f},{:f},{:f},{:d},{:d},{:d},{:d},{:d},{:d},{:f},{:f}".format(
            weldType, frequency, leftAmplitude, rightAmplitude, direction, stopMode, stopTime1, stopTime2, stopTime3, stopTime4, radius, radian)
        if kwargs:
            for key, value in kwargs.items():
                string += ",{}={}".format(key, value)
        string += ")"
        return self.sendRecvMsg(string)

    def WeaveEnd(self):
        return self.sendRecvMsg("WeaveEnd()")

    def WeldArcSpeedStart(self):
        return self.sendRecvMsg("WeldArcSpeedStart()")

    def WeldArcSpeed(self, speed):
        return self.sendRecvMsg("WeldArcSpeed({:f})".format(speed))

    def WeldArcSpeedEnd(self):
        return self.sendRecvMsg("WeldArcSpeedEnd()")

    def WeldWeaveStart(self, weldType, frequency, leftAmplitude, rightAmplitude, direction, stopMode, stopTime1, stopTime2, stopTime3, stopTime4, radius, radian):
        string = "WeldWeaveStart({:d},{:f},{:f},{:f},{:d},{:d},{:d},{:d},{:d},{:d},{:f},{:f})".format(
            weldType, frequency, leftAmplitude, rightAmplitude, direction, stopMode, stopTime1, stopTime2, stopTime3, stopTime4, radius, radian)
        return self.sendRecvMsg(string)


# Feedback interface
# Vendor Dobot TCP/IP API documentation note.


class DobotApiFeedBack(DobotApi):
    def __init__(self, ip, port, *args):
        super().__init__(ip, port, *args)
        self.__MyType = []
        self.last_recv_time = time.perf_counter()
        

    def feedBackData(self):
        'Vendor Dobot TCP/IP API wrapper method.'
        self.socket_dobot.setblocking(True)  # Vendor Dobot TCP/IP API documentation note.
        data = bytes()
        current_recv_time = time.perf_counter() # Vendor Dobot TCP/IP API documentation note.
        temp = self.socket_dobot.recv(144000) # Vendor Dobot TCP/IP API documentation note.
        if len(temp) > 1440:    
            temp = self.socket_dobot.recv(144000)
        #print("get:",len(temp))
        i=0
        if len(temp) < 1440:
            while i < 5 :
                # Vendor Dobot TCP/IP API documentation note.
                temp = self.socket_dobot.recv(144000)
                if len(temp) > 1440:
                    break
                i+=1
            if i >= 5:
                raise Exception('Dobot API message.')
        
        interval = (current_recv_time - self.last_recv_time) * 1000  # Vendor Dobot TCP/IP API documentation note.
        self.last_recv_time = current_recv_time
        #print(f"Time interval since last receive: {interval:.3f} ms")
        
        data = temp[0:1440] # Vendor Dobot TCP/IP API documentation note.
        #print(len(data))
        #print(f"Single element size of MyType: {MyType.itemsize} bytes")
        self.__MyType = None   

        if len(data) == 1440:        
            self.__MyType = np.frombuffer(data, dtype=MyType)

        return self.__MyType
        
