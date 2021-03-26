# -*- coding: utf-8 -*-
import os
import subprocess
import platform
import time
from enum import Enum


class KEYS(Enum):
    #关于键值宏的定义在 KeyEvent.java文件中有定义
    KEYCODE_HOME = 3
    KEYCODE_BACK = 4
    KEYCODE_CALL = 5
    KEYCODE_ENDCALL = 6
    KEYCODE_0 = 7
    KEYCODE_1 = 8
    KEYCODE_2 = 9
    KEYCODE_3 = 10
    KEYCODE_4 = 11
    KEYCODE_5 = 12
    KEYCODE_6 = 13
    KEYCODE_7 = 14
    KEYCODE_8 = 15
    KEYCODE_9 = 16
    KEYCODE_STAR = 17
    KEYCODE_POUND = 18
    KEYCODE_DPAD_UP = 19
    KEYCODE_DPAD_DOWN = 20
    KEYCODE_DPAD_LEFT = 21
    KEYCODE_DPAD_RIGHT = 22
    KEYCODE_DPAD_CENTER = 23

    KEYCODE_VOLUME_UP = 24
    KEYCODE_VOLUME_DOWN = 25
    KEYCODE_POWER = 26

    KEYCODE_CAMERA = 27
    KEYCODE_CLEAR = 28



class auto_adb():
    def __init__(self):
        try:
            adb_path = 'adb'
            subprocess.Popen([adb_path], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            self.adb_path = adb_path
        except OSError:
            if platform.system() == 'Windows':
                adb_path = os.path.join('Tools', "adb", 'adb.exe')
                try:
                    subprocess.Popen(
                        [adb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.adb_path = adb_path
                except OSError:
                    pass
            else:
                try:
                    subprocess.Popen(
                        [adb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except OSError:
                    pass
            print('请安装 ADB 及驱动并配置环境变量')
            print('具体链接: https://github.com/wangshub/wechat_jump_game/wiki')
            exit(1)

    def get_screen(self):
        process = os.popen(self.adb_path + ' shell wm size')
        output = process.read()
        return output

    def run(self, raw_command):
        command = '{} {}'.format(self.adb_path, raw_command)
        process = os.popen(command)
        output = process.read()
        return output

    def test_device(self):
        print('检查设备是否连接...')
        command_list = [self.adb_path, 'devices']
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()
        if output[0].decode('utf8') == 'List of devices attached\n\n':
            print('未找到设备')
            print('adb 输出:')
            for each in output:
                print(each.decode('utf8'))
            return ''
            # exit(1)
        # print('设备已连接')
        # print('adb 输出:')
        outputs=''
        for each in output:
            outputs += "".join(each.decode('utf8'))
        # print(outputs)
        return outputs

    def test_density(self):
        process = os.popen(self.adb_path + ' shell wm density')
        output = process.read()
        return output

    def test_device_detail(self):
        process = os.popen(self.adb_path + ' shell getprop ro.product.device')
        output = process.read()
        return output

    def test_device_os(self):
        process = os.popen(self.adb_path + ' shell getprop ro.build.version.release')
        output = process.read()
        return output

    def adb_path(self):
        return self.adb_path

    def capture_pic(self, img_path):
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        with open(img_path, 'wb') as f:
            f.write(screenshot)

    def wait_adb_connected(self):
        while True:
            output = self.test_device()
            output = output.replace(r'List of devices attached', '')
            lines = output.split('\n')
            for line in lines:
                if len(line) >0 and line.index('device')!= 0:
                    print(line)
                    return line
            time.sleep(1)
        pass

    def auto_input_text(self, text):
        self.run('shell input text '+ text)

    def auto_keypress(self, key):
        self.run('shell input keyevent ' + str(key))

    def auto_tap(self,x, y):
        self.run('shell input tap {} {}'.format(x, y))

    def auto_swipe(self, x1, y1, x2, y2, duration):
        self.run('shell input swipe {} {} {} {} {}'.format(x1, y1, x2, y2, duration))


if __name__ == '__main__':
    adb = auto_adb()
    # adb.capture_pic(r'/home/ai/temp/scr.png')
    # print(adb.run('shell pm list package| grep ziyatech'))
    # adb.wait_adb_connected()
    # adb.auto_tap(1000,1200)
    adb.auto_keypress(KEYS.KEYCODE_POWER.value)
    time.sleep(1)
    adb.auto_swipe(300,1000, 100, 300, 1000)
    time.sleep(1)
    adb.auto_keypress(KEYS.KEYCODE_1.value)
    adb.auto_keypress(KEYS.KEYCODE_2.value)
    adb.auto_keypress(KEYS.KEYCODE_2.value)
    adb.auto_keypress(KEYS.KEYCODE_1.value)
    pass


