import sensor, image, time, tf, json, math
from pyb import Pin

# 按钮引脚定义
BTN_SELECT = Pin('P0', Pin.IN, Pin.PULL_UP)  # 选择（按下确认当前选项）
BTN_UP = Pin('P1', Pin.IN, Pin.PULL_UP)      # 上/确认（注册时拍照）
BTN_DOWN = Pin('P2', Pin.IN, Pin.PULL_UP)    # 下
BTN_BACK = Pin('P3', Pin.IN, Pin.PULL_UP)    # 返回/删除

# 系统配置
class SystemConfig:
    DB_PATH = "user_database.json"
    FACE_MODEL = "face_recognition_model.kmodel"
    LANDMARK_MODEL = "landmark_model.kmodel"
    CASCADE_MODEL = "frontalface"
    MIN_FACE_SIZE = 80
    REGISTRATION_SAMPLES = 5
    SIMILARITY_THRESHOLD = 0.5

# 初始化摄像头
def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_vflip(True)  # 根据安装方向调整
    sensor.set_hmirror(True)  # 根据安装方向调整
    sensor.skip_frames(time = 2000)
    sensor.set_auto_gain(False)  # 关闭自动增益
    sensor.set_auto_whitebal(False)  # 关闭自动白平衡

# 初始化模型
def init_models():
    face_cascade = image.HaarCascade(SystemConfig.CASCADE_MODEL)

    face_id_net = None
    try:
        face_id_net = tf.load(SystemConfig.FACE_MODEL)
        print("人脸识别模型加载成功")
    except Exception as e:
        print(f"人脸识别模型加载失败: {e}")

    landmark_net = None
    try:
        landmark_net = tf.load(SystemConfig.LANDMARK_MODEL)
        print("人脸关键点模型加载成功")
    except Exception as e:
        print(f"人脸关键点模型加载失败: {e}")

    return face_cascade, face_id_net, landmark_net

# 按钮操作类
class ButtonHandler:
    @staticmethod
    def is_pressed(button):
        if button.value() == 0:  # 低电平表示按下
            time.sleep_ms(20)    # 消抖
            if button.value() == 0:
                # 等待释放
                while button.value() == 0:
                    time.sleep_ms(10)
                return True
        return False

    @staticmethod
    def wait_for_any():
        while True:
            if ButtonHandler.is_pressed(BTN_UP) or ButtonHandler.is_pressed(BTN_DOWN) or \
               ButtonHandler.is_pressed(BTN_SELECT) or ButtonHandler.is_pressed(BTN_BACK):
                break
            time.sleep_ms(50)

# 菜单系统
class MenuSystem:
    @staticmethod
    def show(title, options, allow_back=True):
        selected = 0

        while True:
            MenuSystem._clear_screen()
            print("="*30)
            print(title)
            print("="*30)

            # 显示选项列表
            for i, option in enumerate(options):
                prefix = "→ " if i == selected else "   "
                print(f"{prefix}{option}")

            print("="*30)
            if allow_back:
                print("上/下: 选择 | 选择: 确认 | 返回: 后退")
            else:
                print("上/下: 选择 | 选择: 确认")

            # 检测按钮
            if ButtonHandler.is_pressed(BTN_UP):
                selected = (selected - 1) % len(options)
            elif ButtonHandler.is_pressed(BTN_DOWN):
                selected = (selected + 1) % len(options)
            elif ButtonHandler.is_pressed(BTN_SELECT):
                return selected
            elif allow_back and ButtonHandler.is_pressed(BTN_BACK):
                return -1

            time.sleep_ms(100)  # 降低CPU使用率

    @staticmethod
    def _clear_screen():
        print("\033c")

# 人脸处理模块
class FaceProcessor:
    def __init__(self, face_cascade, face_id_net, landmark_net):
        self.face_cascade = face_cascade
        self.face_id_net = face_id_net
        self.landmark_net = landmark_net

    def detect(self, img):
        return img.find_features(self.face_cascade, threshold=0.75, scale_factor=1.25)

    def extract_descriptor(self, img, rect):
        if self.face_id_net is None:
            return None

        face_roi = self._align_face(img, rect) if self.landmark_net else img.copy(roi=rect)
        face_roi = face_roi.resize(96, 96)
        features = self.face_id_net.classify(
            face_roi, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5
        )[0].output()
        return features

    def _align_face(self, img, face_rect):
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

# 用户数据库
class UserDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.users = {}
        self._load()

    def _load(self):
        try:
            with open(self.db_path, "r") as f:
                self.users = json.load(f)
            print(f"已加载 {len(self.users)} 个用户数据")
        except Exception as e:
            self.users = {}
            print(f"用户数据库加载失败: {e}")
            print("将创建新的用户数据库")

    def save(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.users, f)
            return True
        except Exception as e:
            print(f"保存用户数据失败: {e}")
            return False

    def add(self, name, descriptor):
        user_id = str(time.ticks_ms())
        self.users[user_id] = {
            "name": name,
            "descriptor": descriptor,
            "registration_time": str(time.localtime()),
            "samples_count": 1
        }
        return self.save(), user_id

    def delete(self, user_id):
        if user_id in self.users:
            del self.users[user_id]
            return self.save()
        return False

    def get(self, user_id):
        return self.users.get(user_id)

    def list_all(self):
        return list(self.users.items())

# 相似度计算
def cosine_similarity(feat1, feat2):
    dot_product = sum(a * b for a, b in zip(feat1, feat2))
    norm_a = sum(a * a for a in feat1) ** 0.5
    norm_b = sum(b * b for b in feat2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# 活体检测
def liveness_detection(face_processor, timeout_ms=5000):
    start_time = time.ticks_ms()
    eyes_detected = False
    eyes_closed = False

    while time.ticks_diff(time.ticks_ms(), start_time) < timeout_ms:
        img = sensor.snapshot()
        faces = face_processor.detect(img)

        if faces:
            face = faces[0]
            eyes = img.find_features(image.HaarCascade("eye"), threshold=0.75, scale_factor=1.25, roi=face)

            if eyes and not eyes_detected:
                eyes_detected = True
            elif eyes_detected and not eyes:
                eyes_closed = True
                break

        if ButtonHandler.is_pressed(BTN_SELECT):
            break

        time.sleep_ms(100)

    return eyes_detected and eyes_closed

# 人脸识别系统
class FaceRecognitionSystem:
    def __init__(self):
        init_camera()
        face_cascade, face_id_net, landmark_net = init_models()

        self.face_processor = FaceProcessor(face_cascade, face_id_net, landmark_net)
        self.user_db = UserDatabase(SystemConfig.DB_PATH)

    def run(self):
        print("系统初始化完成")
        while True:
            choice = MenuSystem.show("OpenMV人脸识别系统", [
                "开始识别", "注册新人脸", "管理用户", "退出"
            ])

            if choice == 0:
                self._recognition_mode()
            elif choice == 1:
                self._registration_mode()
            elif choice == 2:
                self._manage_users()
            elif choice == 3:
                print("系统已关闭")
                break

    def _recognition_mode(self):
        MenuSystem._clear_screen()
        print("开始人脸识别，按选择键退出")
        print("="*30)

        while True:
            img = sensor.snapshot()
            faces = self.face_processor.detect(img)

            if faces:
                face = faces[0]
                img.draw_rectangle(face)

                descriptor = self.face_processor.extract_descriptor(img, face)
                face_id = self._recognize_face(descriptor)

                if face_id:
                    user_data = self.user_db.get(face_id)
                    if user_data:
                        img.draw_string(face[0], face[1]-15, f"{user_data['name']}", color=(0, 255, 0))
                        print(f"识别结果: {user_data['name']}")
                else:
                    img.draw_string(face[0], face[1]-15, "未知人脸", color=(255, 0, 0))
                    print("识别结果: 未知人脸")

            if ButtonHandler.is_pressed(BTN_SELECT):
                print("已退出识别模式")
                time.sleep_ms(500)
                break

            time.sleep_ms(100)

    def _registration_mode(self):
        print("进入人脸注册模式")
        name = self._chinese_input("输入用户姓名")
        if not name:
            print("注册已取消")
            return

        print(f"用户姓名: {name}")
        print("请将人脸对准摄像头")
        print("按上键拍照，按选择键完成注册")

        samples = []
        while len(samples) < SystemConfig.REGISTRATION_SAMPLES:
            img = sensor.snapshot()
            faces = self.face_processor.detect(img)

            if faces:
                face = faces[0]
                img.draw_rectangle(face)
                img.draw_string(face[0], face[1]-10, f"样本 {len(samples)+1}/{SystemConfig.REGISTRATION_SAMPLES}")

                if ButtonHandler.is_pressed(BTN_UP):
                    if face[2] < SystemConfig.MIN_FACE_SIZE or face[3] < SystemConfig.MIN_FACE_SIZE:
                        print("人脸距离太远，请靠近摄像头")
                        time.sleep_ms(1000)
                        continue

                    descriptor = self.face_processor.extract_descriptor(img, face)
                    if descriptor:
                        samples.append(descriptor)
                        print(f"已拍摄样本 {len(samples)}/{SystemConfig.REGISTRATION_SAMPLES}")
                    else:
                        print("未提取到有效特征，请调整角度")
                        time.sleep_ms(1000)

            if ButtonHandler.is_pressed(BTN_SELECT) and samples:
                break

        if not samples:
            print("注册失败：未采集到有效样本")
            return

        print("请眨眼进行活体检测...")
        if not liveness_detection(self.face_processor):
            print("活体检测失败")
            return

        avg_descriptor = [sum(features[i] for features in samples) / len(samples) for i in range(len(samples[0]))]
        success, user_id = self.user_db.add(name, avg_descriptor)

        if success:
            print(f"成功注册用户: {name}")
        else:
            print("注册失败")

    def _manage_users(self):
        while True:
            MenuSystem._clear_screen()
            print("="*30)
            print("用户管理")
            print("="*30)

            users = self.user_db.list_all()
            if not users:
                print("暂无注册用户")
                print("="*30)
                print("按选择键返回")
                ButtonHandler.wait_for_any()
                return

            # 显示用户列表
            options = [f"{user_data['name']} ({user_id})" for user_id, user_data in users]
            options.append("返回")

            choice = MenuSystem.show("用户管理", options)
            if choice == -1 or choice == len(options) - 1:
                return

            user_id = users[choice][0]
            sub_choice = MenuSystem.show(f"用户: {self.user_db.get(user_id)['name']}", [
                "查看详情", "删除用户", "返回"
            ])

            if sub_choice == 0:
                self._view_user_details(user_id)
            elif sub_choice == 1:
                self._delete_user(user_id)

    def _view_user_details(self, user_id):
        user_data = self.user_db.get(user_id)
        if not user_data:
            print("用户不存在")
            time.sleep_ms(1000)
            return

        MenuSystem._clear_screen()
        print("="*30)
        print(f"用户详情: {user_data['name']}")
        print("="*30)
        print(f"用户ID: {user_id}")
        print(f"注册时间: {user_data['registration_time']}")
        print(f"样本数量: {user_data['samples_count']}")
        print("="*30)
        print("按任意键返回")
        ButtonHandler.wait_for_any()

    def _delete_user(self, user_id):
        user_data = self.user_db.get(user_id)
        if not user_data:
            print("用户不存在")
            time.sleep_ms(1000)
            return

        confirm = MenuSystem.show(f"确认删除用户: {user_data['name']}?", ["确认", "取消"], False)
        if confirm == 0:
            if self.user_db.delete(user_id):
                print(f"已删除用户: {user_data['name']}")
            else:
                print("删除用户失败")
            time.sleep_ms(1000)

    def _recognize_face(self, descriptor):
        if descriptor is None:
            return None

        best_match_id = None
        highest_similarity = SystemConfig.SIMILARITY_THRESHOLD

        for user_id, user_data in self.user_db.list_all():
            similarity = cosine_similarity(descriptor, user_data["descriptor"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = user_id

        return best_match_id

    def _chinese_input(self, title, max_length=8):
        chinese_chars = [
            "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨",
            "朱秦尤许何吕施张孔曹严华金魏陶姜",
            "戚谢邹喻柏水窦章云苏潘葛奚范彭郎",
            "鲁韦昌马苗凤花方俞任袁柳酆鲍史唐",
            "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常",
            "乐于时傅皮卞齐康伍余元卜顾孟平黄",
            "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧",
            "计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
        ]

        current_text = ""
        page = 0
        selected = 0

        while True:
            MenuSystem._clear_screen()
            print("="*30)
            print(title)
            print("="*30)
            print(f"当前输入: {current_text}")
            print("-"*30)

            if page < len(chinese_chars):
                chars = chinese_chars[page]
                for i in range(0, len(chars), 4):
                    line = ""
                    for j in range(4):
                        if i+j < len(chars):
                            prefix = "[" if i+j == selected else " "
                            suffix = "]" if i+j == selected else " "
                            line += f"{prefix}{chars[i+j]}{suffix} "
                    print(line)

            print("-"*30)
            print("上/下: 选择 | 选择: 添加 | 返回: 删除 | 下页: 确认")

            if ButtonHandler.is_pressed(BTN_UP):
                selected = (selected - 1) % len(chinese_chars[page])
            elif ButtonHandler.is_pressed(BTN_DOWN):
                selected = (selected + 1) % len(chinese_chars[page])
            elif ButtonHandler.is_pressed(BTN_SELECT):
                if page < len(chinese_chars) and len(current_text) < max_length:
                    current_text += chinese_chars[page][selected]
            elif ButtonHandler.is_pressed(BTN_BACK):
                if current_text:
                    current_text = current_text[:-1]
                else:
                    return ""
            elif ButtonHandler.is_pressed(BTN_UP) and ButtonHandler.is_pressed(BTN_DOWN):
                if current_text:
                    return current_text
                else:
                    print("姓名不能为空！")
                    time.sleep_ms(1000)

            time.sleep_ms(100)

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
