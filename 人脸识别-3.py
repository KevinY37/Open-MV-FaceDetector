import sensor, image, time, tf, json, math
from pyb import Pin

BTN_UP = Pin('P1', Pin.IN, Pin.PULL_UP)
BTN_DOWN = Pin('P2', Pin.IN, Pin.PULL_UP)
BTN_SELECT = Pin('P0', Pin.IN, Pin.PULL_UP)
BTN_BACK = Pin('P3', Pin.IN, Pin.PULL_UP)

class Config:
    """系统配置类，集中管理所有参数"""
    DB_PATH = "user_database.json"
    FACE_MODEL = "face_recognition_model.kmodel"
    LANDMARK_MODEL = "landmark_model.kmodel"
    CASCADE_MODEL = "frontalface"
    DISPLAY_WIDTH = 320
    DISPLAY_HEIGHT = 240
    MIN_FACE_SIZE = 80
    REGISTRATION_SAMPLES = 5
    SIMILARITY_THRESHOLD = 0.5
    LIVENESS_TIMEOUT = 5000
    BUTTON_DEBOUNCE = 20
    MENU_UPDATE_DELAY = 100

class Button:
    """按钮处理类，支持长按、短按等操作"""
    def __init__(self, pin, debounce=Config.BUTTON_DEBOUNCE):
        self.pin = pin
        self.debounce = debounce
        self.last_state = 1
        self.pressed_time = 0
        self.long_press_threshold = 1000  # 长按阈值(ms)

    def is_pressed(self):
        """检测短按"""
        state = self.pin.value()
        if state == 0 and self.last_state == 1:
            time.sleep_ms(self.debounce)
            if self.pin.value() == 0:
                self.last_state = 0
                self.pressed_time = time.ticks_ms()
                return True
        elif state == 1 and self.last_state == 0:
            self.last_state = 1
        return False

    def is_long_pressed(self):
        """检测长按"""
        state = self.pin.value()
        if state == 0 and self.last_state == 1:
            time.sleep_ms(self.debounce)
            if self.pin.value() == 0:
                self.last_state = 0
                self.pressed_time = time.ticks_ms()
        elif state == 0 and self.last_state == 0:
            if time.ticks_diff(time.ticks_ms(), self.pressed_time) > self.long_press_threshold:
                self.last_state = 1  # 避免重复触发
                return True
        elif state == 1 and self.last_state == 0:
            self.last_state = 1
        return False

class Display:
    """显示处理类，统一管理屏幕输出"""
    @staticmethod
    def clear():
        print("\033c")

    @staticmethod
    def show_title(text):
        print("="*30)
        print(text)
        print("="*30)

    @staticmethod
    def show_message(text, duration=1000):
        Display.clear()
        Display.show_title("系统提示")
        print(text)
        time.sleep_ms(duration)

    @staticmethod
    def draw_face_info(img, face_rect, text, color=(255, 255, 255)):
        img.draw_rectangle(face_rect)
        img.draw_string(face_rect[0], face_rect[1]-15, text, color=color)

class Menu:
    """菜单系统类，支持多级菜单和自定义操作"""
    def __init__(self, title, options, action_map=None):
        self.title = title
        self.options = options
        self.action_map = action_map or {}
        self.selected = 0

    def show(self, buttons):
        """显示菜单并处理用户输入"""
        while True:
            Display.clear()
            Display.show_title(self.title)

            for i, option in enumerate(self.options):
                prefix = "→ " if i == self.selected else "   "
                print(f"{prefix}{option}")

            print("="*30)
            print("上: 上一项 | 下: 下一项 | 选择: 确认 | 返回: 后退")

            if buttons["up"].is_pressed():
                self.selected = (self.selected - 1) % len(self.options)
            elif buttons["down"].is_pressed():
                self.selected = (self.selected + 1) % len(self.options)
            elif buttons["select"].is_pressed():
                if self.selected in self.action_map:
                    return self.action_map[self.selected]()
                else:
                    return self.selected
            elif buttons["back"].is_pressed():
                return -1

            time.sleep_ms(Config.MENU_UPDATE_DELAY)

class FaceDetector:
    """人脸检测与特征提取类"""
    def __init__(self, cascade_model, face_model, landmark_model=None):
        self.face_cascade = image.HaarCascade(cascade_model)
        self.face_id_net = tf.load(face_model) if face_model else None
        self.landmark_net = tf.load(landmark_model) if landmark_model else None

    def detect_faces(self, img):
        """检测图像中的人脸"""
        return img.find_features(self.face_cascade, threshold=0.75, scale_factor=1.25)

    def extract_features(self, img, face_rect):
        """提取人脸特征向量"""
        if not self.face_id_net:
            return None

        face_roi = self._align_face(img, face_rect) if self.landmark_net else img.copy(roi=face_rect)
        face_roi = face_roi.resize(96, 96)
        return self.face_id_net.classify(
            face_roi, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5
        )[0].output()

    def _align_face(self, img, face_rect):
        """人脸对齐处理"""
        face_roi = img.copy(roi=face_rect)
        landmarks = self.landmark_net.classify(face_roi)[0].output()

        left_eye = (landmarks[0] * face_rect[2] + face_rect[0],
                    landmarks[1] * face_rect[3] + face_rect[1])
        right_eye = (landmarks[2] * face_rect[2] + face_rect[0],
                     landmarks[3] * face_rect[3] + face_rect[1])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.atan2(dy, dx) * 180 / math.pi

        img_aligned = img.copy()
        img_aligned.rotate(angle)

        cx, cy = (face_rect[0] + face_rect[2]//2, face_rect[1] + face_rect[3]//2)
        w, h = face_rect[2], face_rect[3]
        return img_aligned.copy(roi=(cx - w//2, cy - h//2, w, h))

class UserManager:
    """用户管理类，处理用户数据的增删改查"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.users = self._load_db()

    def _load_db(self):
        """加载用户数据库"""
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_db(self):
        """保存用户数据库"""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.users, f)
            return True
        except Exception as e:
            print(f"保存数据库失败: {e}")
            return False

    def add_user(self, name, features):
        """添加新用户"""
        user_id = str(time.ticks_ms())
        self.users[user_id] = {
            "name": name,
            "features": features,
            "registered_at": str(time.localtime()),
            "samples_count": 1
        }
        return self.save_db(), user_id

    def delete_user(self, user_id):
        """删除用户"""
        if user_id in self.users:
            del self.users[user_id]
            return self.save_db()
        return False

    def get_user(self, user_id):
        """获取用户信息"""
        return self.users.get(user_id)

    def get_all_users(self):
        """获取所有用户"""
        return list(self.users.items())

    def find_user_by_features(self, features, threshold=Config.SIMILARITY_THRESHOLD):
        """通过特征查找用户"""
        if not features:
            return None

        best_match = None
        highest_similarity = threshold

        for user_id, user_data in self.users.items():
            similarity = self._cosine_similarity(features, user_data["features"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user_id

        return best_match

    @staticmethod
    def _cosine_similarity(feat1, feat2):
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(feat1, feat2))
        norm_a = sum(a * a for a in feat1) ** 0.5
        norm_b = sum(b * b for b in feat2) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

class ChineseInput:
    """中文输入法类"""
    CHAR_SETS = [
        "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨",
        "朱秦尤许何吕施张孔曹严华金魏陶姜",
        "戚谢邹喻柏水窦章云苏潘葛奚范彭郎",
        "鲁韦昌马苗凤花方俞任袁柳酆鲍史唐",
        "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常",
        "乐于时傅皮卞齐康伍余元卜顾孟平黄",
        "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧",
        "计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    ]

    def __init__(self, title="输入姓名", max_length=8):
        self.title = title
        self.max_length = max_length

    def input(self, buttons):
        """获取用户输入的中文字符串"""
        current_text = ""
        page = 0
        selected = 0

        while True:
            Display.clear()
            Display.show_title(self.title)
            print(f"当前输入: {current_text}")
            print("-"*30)

            if page < len(self.CHAR_SETS):
                chars = self.CHAR_SETS[page]
                for i in range(0, len(chars), 4):
                    line = ""
                    for j in range(4):
                        if i+j < len(chars):
                            prefix = "[" if i+j == selected else " "
                            suffix = "]" if i+j == selected else " "
                            line += f"{prefix}{chars[i+j]}{suffix} "
                    print(line)

            print("-"*30)
            print("上: 上一项 | 下: 下一项 | 选择: 添加 | 返回: 删除 | 长按上+下: 确认")

            if buttons["up"].is_pressed():
                selected = (selected - 1) % len(self.CHAR_SETS[page])
            elif buttons["down"].is_pressed():
                selected = (selected + 1) % len(self.CHAR_SETS[page])
            elif buttons["select"].is_pressed():
                if page < len(self.CHAR_SETS) and len(current_text) < self.max_length:
                    current_text += self.CHAR_SETS[page][selected]
            elif buttons["back"].is_pressed():
                if current_text:
                    current_text = current_text[:-1]
                else:
                    return ""  # 取消输入
            elif buttons["up"].is_long_pressed() and buttons["down"].is_long_pressed():
                if current_text:
                    return current_text
                else:
                    Display.show_message("姓名不能为空！")

            time.sleep_ms(Config.MENU_UPDATE_DELAY)

class FaceRecognitionApp:
    """人脸识别应用主类"""
    def __init__(self):
        # 初始化硬件
        self._init_hardware()

        # 初始化组件
        self.buttons = {
            "up": Button(BTN_UP),
            "down": Button(BTN_DOWN),
            "select": Button(BTN_SELECT),
            "back": Button(BTN_BACK)
        }

        self.face_detector = FaceDetector(
            Config.CASCADE_MODEL,
            Config.FACE_MODEL,
            Config.LANDMARK_MODEL
        )

        self.user_manager = UserManager(Config.DB_PATH)

    def _init_hardware(self):
        """初始化摄像头"""
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_vflip(True)
        sensor.set_hmirror(True)
        sensor.skip_frames(time=2000)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)

    def run(self):
        """运行应用主循环"""
        Display.show_message("系统初始化完成")

        main_menu = Menu("人脸识别系统", [
            "开始识别", "注册新人脸", "管理用户", "系统设置", "关于", "退出"
        ], {
            0: self._recognition_mode,
            1: self._registration_mode,
            2: self._manage_users,
            3: self._system_settings,
            4: self._show_about,
            5: self._exit_system
        })

        while True:
            main_menu.show(self.buttons)

    def _recognition_mode(self):
        """人脸识别模式"""
        Display.show_message("开始人脸识别，按返回键退出")

        while True:
            img = sensor.snapshot()
            faces = self.face_detector.detect_faces(img)

            if faces:
                for face in faces:
                    features = self.face_detector.extract_features(img, face)
                    user_id = self.user_manager.find_user_by_features(features)

                    if user_id:
                        user = self.user_manager.get_user(user_id)
                        Display.draw_face_info(img, face, user["name"], (0, 255, 0))
                    else:
                        Display.draw_face_info(img, face, "未知人脸", (255, 0, 0))

            if self.buttons["back"].is_pressed():
                break

    def _registration_mode(self):
        """人脸注册模式"""
        Display.show_message("进入人脸注册模式")

        # 获取用户姓名
        inputer = ChineseInput("输入用户姓名")
        name = inputer.input(self.buttons)

        if not name:
            Display.show_message("注册已取消")
            return

        Display.show_message(f"用户姓名: {name}\n请将人脸对准摄像头\n按上键拍照，按返回键完成注册")

        samples = []
        while len(samples) < Config.REGISTRATION_SAMPLES:
            img = sensor.snapshot()
            faces = self.face_detector.detect_faces(img)

            if faces:
                face = max(faces, key=lambda f: f[2] * f[3])  # 选择最大的人脸
                img.draw_rectangle(face)
                img.draw_string(face[0], face[1]-10,
                               f"样本 {len(samples)+1}/{Config.REGISTRATION_SAMPLES}")

                if self.buttons["up"].is_pressed():
                    if face[2] < Config.MIN_FACE_SIZE or face[3] < Config.MIN_FACE_SIZE:
                        Display.show_message("人脸距离太远，请靠近摄像头")
                        continue

                    features = self.face_detector.extract_features(img, face)
                    if features:
                        samples.append(features)
                        Display.show_message(f"已拍摄样本 {len(samples)}/{Config.REGISTRATION_SAMPLES}")
                    else:
                        Display.show_message("未提取到有效特征，请调整角度")

            if self.buttons["back"].is_pressed() and samples:
                break

        if not samples:
            Display.show_message("注册失败：未采集到有效样本")
            return

        # 活体检测
        Display.show_message("请眨眼进行活体检测...")
        if not self._liveness_detection():
            Display.show_message("活体检测失败")
            return

        # 计算平均特征
        avg_features = [sum(f[i] for f in samples) / len(samples) for i in range(len(samples[0]))]

        # 保存用户
        success, user_id = self.user_manager.add_user(name, avg_features)

        if success:
            Display.show_message(f"成功注册用户: {name}")
        else:
            Display.show_message("注册失败")

    def _liveness_detection(self):
        """活体检测（眨眼检测）"""
        start_time = time.ticks_ms()
        eyes_detected = False
        eyes_closed = False

        while time.ticks_diff(time.ticks_ms(), start_time) < Config.LIVENESS_TIMEOUT:
            img = sensor.snapshot()
            faces = self.face_detector.detect_faces(img)

            if faces:
                face = faces[0]
                eyes = img.find_features(image.HaarCascade("eye"), threshold=0.75,
                                        scale_factor=1.25, roi=face)

                if eyes and not eyes_detected:
                    eyes_detected = True
                elif eyes_detected and not eyes:
                    eyes_closed = True
                    break

            if self.buttons["back"].is_pressed():
                break

            time.sleep_ms(100)

        return eyes_detected and eyes_closed

    def _manage_users(self):
        """用户管理模式"""
        while True:
            users = self.user_manager.get_all_users()

            if not users:
                Display.show_message("暂无注册用户")
                return

            user_options = [f"{user['name']} ({user_id})" for user_id, user in users]
            user_options.append("返回")

            user_menu = Menu("用户管理", user_options)
            choice = user_menu.show(self.buttons)

            if choice == -1 or choice == len(user_options) - 1:
                return

            user_id = users[choice][0]
            self._manage_user_details(user_id)

    def _manage_user_details(self, user_id):
        """管理用户详情"""
        user = self.user_manager.get_user(user_id)
        if not user:
            Display.show_message("用户不存在")
            return

        options = [
            "查看详情", "重命名", "重新采集人脸", "删除用户", "返回"
        ]

        user_menu = Menu(f"用户: {user['name']}", options, {
            0: lambda: self._view_user_details(user_id),
            1: lambda: self._rename_user(user_id),
            2: lambda: self._recollect_user_features(user_id),
            3: lambda: self._delete_user(user_id),
            4: lambda: -1
        })

        user_menu.show(self.buttons)

    def _view_user_details(self, user_id):
        """查看用户详情"""
        user = self.user_manager.get_user(user_id)
        if not user:
            Display.show_message("用户不存在")
            return

        Display.clear()
        Display.show_title(f"用户详情: {user['name']}")
        print(f"用户ID: {user_id}")
        print(f"注册时间: {user['registered_at']}")
        print(f"样本数量: {user['samples_count']}")
        print("="*30)
        print("按任意键返回")

        while not any(btn.is_pressed() for btn in self.buttons.values()):
            time.sleep_ms(100)

    def _rename_user(self, user_id):
        """重命名用户"""
        user = self.user_manager.get_user(user_id)
        if not user:
            Display.show_message("用户不存在")
            return

        inputer = ChineseInput(f"重命名: {user['name']}")
        new_name = inputer.input(self.buttons)

        if new_name and new_name != user['name']:
            user['name'] = new_name
            if self.user_manager.save_db():
                Display.show_message(f"已重命名为: {new_name}")
            else:
                Display.show_message("保存失败")

    def _recollect_user_features(self, user_id):
        """重新采集用户人脸特征"""
        user = self.user_manager.get_user(user_id)
        if not user:
            Display.show_message("用户不存在")
            return

        Display.show_message(f"为用户 {user['name']} 重新采集人脸\n请将人脸对准摄像头\n按上键拍照，按返回键完成")

        samples = []
        while len(samples) < Config.REGISTRATION_SAMPLES:
            img = sensor.snapshot()
            faces = self.face_detector.detect_faces(img)

            if faces:
                face = max(faces, key=lambda f: f[2] * f[3])
                img.draw_rectangle(face)
                img.draw_string(face[0], face[1]-10,
                               f"样本 {len(samples)+1}/{Config.REGISTRATION_SAMPLES}")

                if self.buttons["up"].is_pressed():
                    if face[2] < Config.MIN_FACE_SIZE or face[3] < Config.MIN_FACE_SIZE:
                        Display.show_message("人脸距离太远，请靠近摄像头")
                        continue

                    features = self.face_detector.extract_features(img, face)
                    if features:
                        samples.append(features)
                        Display.show_message(f"已拍摄样本 {len(samples)}/{Config.REGISTRATION_SAMPLES}")
                    else:
                        Display.show_message("未提取到有效特征，请调整角度")

            if self.buttons["back"].is_pressed() and samples:
                break

        if not samples:
            Display.show_message("采集失败：未获取到有效样本")
            return

        avg_features = [sum(f[i] for f in samples) / len(samples) for i in range(len(samples[0]))]
        user['features'] = avg_features
        user['samples_count'] = len(samples)

        if self.user_manager.save_db():
            Display.show_message("人脸特征已更新")
        else:
            Display.show_message("保存失败")

    def _delete_user(self, user_id):
        """删除用户"""
        user = self.user_manager.get_user(user_id)
        if not user:
            Display.show_message("用户不存在")
            return

        confirm_menu = Menu(f"确认删除用户: {user['name']}?", ["确认", "取消"])
        choice = confirm_menu.show(self.buttons)

        if choice == 0:
            if self.user_manager.delete_user(user_id):
                Display.show_message(f"已删除用户: {user['name']}")
            else:
                Display.show_message("删除失败")

    def _system_settings(self):
        """系统设置"""
        settings_menu = Menu("系统设置", [
            "识别阈值: {:.1f}".format(Config.SIMILARITY_THRESHOLD),
            "活体检测: {}".format("开启" if Config.LIVENESS_TIMEOUT > 0 else "关闭"),
            "重置数据库", "返回"
        ], {
            0: self._adjust_threshold,
            1: self._toggle_liveness_detection,
            2: self._reset_database,
            3: lambda: -1
        })

        settings_menu.show(self.buttons)

    def _adjust_threshold(self):
        """调整识别阈值"""
        current_threshold = Config.SIMILARITY_THRESHOLD
        while True:
            Display.clear()
            Display.show_title("调整识别阈值")
            print(f"当前阈值: {current_threshold:.1f}")
            print("="*30)
            print("上: 增加 0.1 | 下: 减少 0.1")
            print("选择: 确认 | 返回: 取消")

            if self.buttons["up"].is_pressed():
                current_threshold = min(1.0, current_threshold + 0.1)
            elif self.buttons["down"].is_pressed():
                current_threshold = max(0.1, current_threshold - 0.1)
            elif self.buttons["select"].is_pressed():
                Config.SIMILARITY_THRESHOLD = current_threshold
                Display.show_message(f"已设置识别阈值为: {current_threshold:.1f}")
                break
            elif self.buttons["back"].is_pressed():
                break

            time.sleep_ms(Config.MENU_UPDATE_DELAY)

    def _toggle_liveness_detection(self):
        """切换活体检测状态"""
        if Config.LIVENESS_TIMEOUT > 0:
            Config.LIVENESS_TIMEOUT = 0
            Display.show_message("活体检测已关闭")
        else:
            Config.LIVENESS_TIMEOUT = 5000
            Display.show_message("活体检测已开启")

    def _reset_database(self):
        """重置数据库"""
        confirm_menu = Menu("确认重置数据库?", ["确认", "取消"])
        choice = confirm_menu.show(self.buttons)

        if choice == 0:
            self.user_manager.users = {}
            if self.user_manager.save_db():
                Display.show_message("数据库已重置")
            else:
                Display.show_message("重置失败")

    def _show_about(self):
        """显示关于信息"""
        Display.clear()
        Display.show_title("关于")
        print("OpenMV人脸识别系统")
        print("版本: 1.0.0")
        print("="*30)
        print("按任意键返回")

        while not any(btn.is_pressed() for btn in self.buttons.values()):
            time.sleep_ms(100)

    def _exit_system(self):
        """退出系统"""
        Display.show_message("系统已关闭")
        return -1

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
