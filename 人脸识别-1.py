import sensor, image, time, tf, json, math
from pyb import Pin

# 按钮引脚定义
BTN_SELECT = Pin('P0', Pin.IN, Pin.PULL_UP)  # 选择（按下确认当前选项）
BTN_UP = Pin('P1', Pin.IN, Pin.PULL_UP)      # 上/确认（注册时拍照）
BTN_DOWN = Pin('P2', Pin.IN, Pin.PULL_UP)    # 下
BTN_BACK = Pin('P3', Pin.IN, Pin.PULL_UP)    # 返回/删除

# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_vflip(True)  # 根据安装方向调整
sensor.set_hmirror(True)  # 根据安装方向调整
sensor.skip_frames(time = 2000)
sensor.set_auto_gain(False)  # 关闭自动增益
sensor.set_auto_whitebal(False)  # 关闭自动白平衡

# 模型路径
face_cascade_path = "frontalface"  # OpenMV内置人脸检测模型
face_id_model_path = "face_recognition_model.kmodel"  # 人脸识别模型
landmark_model_path = "landmark_model.kmodel"  # 人脸关键点模型

# 初始化模型
face_cascade = image.HaarCascade(face_cascade_path)

try:
    face_id_net = tf.load(face_id_model_path)
    print("人脸识别模型加载成功")
except Exception as e:
    face_id_net = None
    print(f"人脸识别模型加载失败: {e}")

try:
    landmark_net = tf.load(landmark_model_path)
    print("人脸关键点模型加载成功")
except Exception as e:
    landmark_net = None
    print(f"人脸关键点模型加载失败: {e}")

# 用户数据库路径
USER_DB_PATH = "user_database.json"

# 加载已有用户数据
try:
    with open(USER_DB_PATH, "r") as f:
        user_db = json.load(f)
    print(f"已加载 {len(user_db)} 个用户数据")
except Exception as e:
    user_db = {}
    print(f"用户数据库加载失败: {e}")
    print("将创建新的用户数据库")

# 按钮检测函数（消抖）
def is_button_pressed(button):
    if button.value() == 0:  # 低电平表示按下
        time.sleep_ms(20)    # 消抖
        if button.value() == 0:
            # 等待释放
            while button.value() == 0:
                time.sleep_ms(10)
            return True
    return False

# 等待任意按钮按下
def wait_for_any_button():
    while True:
        if is_button_pressed(BTN_UP) or is_button_pressed(BTN_DOWN) or \
           is_button_pressed(BTN_SELECT) or is_button_pressed(BTN_BACK):
            break
        time.sleep_ms(50)

# 按钮菜单选择器
def button_menu(title, options, allow_back=True):
    selected = 0

    while True:
        # 清屏
        print("\033c")
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
        if is_button_pressed(BTN_UP):
            selected = (selected - 1) % len(options)
        elif is_button_pressed(BTN_DOWN):
            selected = (selected + 1) % len(options)
        elif is_button_pressed(BTN_SELECT):
            return selected
        elif allow_back and is_button_pressed(BTN_BACK):
            return -1

        time.sleep_ms(100)  # 降低CPU使用率

# 中文输入界面
def chinese_input(title, max_length=8):
    # 简化的中文输入法，使用数字键选择常用汉字
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
        print("\033c")
        print("="*30)
        print(title)
        print("="*30)
        print(f"当前输入: {current_text}")
        print("-"*30)

        # 显示当前页的汉字
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

        # 检测按钮
        if is_button_pressed(BTN_UP):
            selected = (selected - 1) % len(chinese_chars[page])
        elif is_button_pressed(BTN_DOWN):
            selected = (selected + 1) % len(chinese_chars[page])
        elif is_button_pressed(BTN_SELECT):
            if page < len(chinese_chars) and len(current_text) < max_length:
                current_text += chinese_chars[page][selected]
        elif is_button_pressed(BTN_BACK):
            if current_text:
                current_text = current_text[:-1]
            else:
                return ""  # 返回空表示取消
        elif is_button_pressed(BTN_UP) and is_button_pressed(BTN_DOWN):  # 同时按下上和下退出
            if current_text:
                return current_text
            else:
                print("姓名不能为空！")
                time.sleep_ms(1000)

        time.sleep_ms(100)

# 人脸对齐函数
def align_face(img, face_rect):
    if landmark_net is None:
        return img.copy(roi=face_rect)

    # 检测人脸关键点
    face_roi = img.copy(roi=face_rect)
    landmarks = landmark_net.classify(face_roi)[0].output()

    # 提取左右眼和嘴的关键点
    left_eye = (landmarks[0] * face_rect[2] + face_rect[0],
                landmarks[1] * face_rect[3] + face_rect[1])
    right_eye = (landmarks[2] * face_rect[2] + face_rect[0],
                 landmarks[3] * face_rect[3] + face_rect[1])

    # 计算旋转角度
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.atan2(dy, dx) * 180 / math.pi

    # 旋转图像使眼睛水平
    img_aligned = img.copy()
    img_aligned.rotate(angle)

    # 重新计算对齐后的人脸区域
    cx, cy = (face_rect[0] + face_rect[2]//2, face_rect[1] + face_rect[3]//2)
    w, h = face_rect[2], face_rect[3]
    new_rect = (cx - w//2, cy - h//2, w, h)

    return img_aligned.copy(roi=new_rect)

# 人脸特征提取函数
def extract_face_descriptor(img, rect):
    if face_id_net is None:
        return None

    # 人脸对齐（如果有关键点模型）
    if landmark_net:
        face_roi = align_face(img, rect)
    else:
        face_roi = img.copy(roi=rect)

    # 调整人脸区域大小为模型输入尺寸
    face_roi = face_roi.resize(96, 96)

    # 提取特征
    features = face_id_net.classify(face_roi, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5)[0].output()
    return features

# 计算余弦相似度
def cosine_similarity(feat1, feat2):
    dot_product = sum(a * b for a, b in zip(feat1, feat2))
    norm_a = sum(a * a for a in feat1) ** 0.5
    norm_b = sum(b * b for b in feat2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# 人脸比对函数
def recognize_face(descriptor, threshold=0.5):
    if descriptor is None:
        return None

    best_match_id = None
    highest_similarity = threshold  # 低于阈值则认为是未知人脸

    for user_id, user_data in user_db.items():
        known_descriptor = user_data["descriptor"]
        similarity = cosine_similarity(descriptor, known_descriptor)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_id = user_id

    return best_match_id

# 简单的活体检测（眨眼检测）
def liveness_detection(img, face_rect, timeout_ms=5000):
    start_time = time.ticks_ms()
    eyes_detected = False
    eyes_closed = False

    while time.ticks_diff(time.ticks_ms(), start_time) < timeout_ms:
        img = sensor.snapshot()
        faces = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

        if faces:
            face = faces[0]
            # 简化的眨眼检测：检测到眼睛然后眼睛消失表示眨眼
            eyes = img.find_features(image.HaarCascade("eye"), threshold=0.75, scale_factor=1.25, roi=face)

            if eyes and not eyes_detected:
                eyes_detected = True
            elif eyes_detected and not eyes:
                eyes_closed = True
                break

        if is_button_pressed(BTN_SELECT):
            break

        time.sleep_ms(100)

    return eyes_detected and eyes_closed

# 人脸注册模式
def registration_mode():
    print("进入人脸注册模式")

    # 获取用户姓名
    name = chinese_input("输入用户姓名")
    if not name:
        print("注册已取消")
        return False

    print(f"用户姓名: {name}")
    print("请将人脸对准摄像头")
    print("按上键拍照，按选择键完成注册")

    samples = []
    while len(samples) < 5:  # 采集5张样本
        img = sensor.snapshot()
        faces = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

        if faces:
            face = faces[0]
            img.draw_rectangle(face)
            img.draw_string(face[0], face[1]-10, f"样本 {len(samples)+1}/5")

            if is_button_pressed(BTN_UP):
                # 简单的质量检测
                if face[2] < 80 or face[3] < 80:  # 人脸太小
                    print("人脸距离太远，请靠近摄像头")
                    time.sleep_ms(1000)
                    continue

                descriptor = extract_face_descriptor(img, face)
                if descriptor:
                    samples.append(descriptor)
                    print(f"已拍摄样本 {len(samples)}/5")
                else:
                    print("未提取到有效特征，请调整角度")
                    time.sleep_ms(1000)

        if is_button_pressed(BTN_SELECT) and samples:
            break

    if not samples:
        print("注册失败：未采集到有效样本")
        return False

    # 活体检测
    print("请眨眼进行活体检测...")
    if not liveness_detection(img, face):
        print("活体检测失败")
        return False

    # 取平均特征作为注册特征
    avg_descriptor = [sum(features[i] for features in samples) / len(samples) for i in range(len(samples[0]))]

    # 生成唯一用户ID
    user_id = str(time.ticks_ms())  # 使用时间戳作为ID

    # 保存到数据库
    user_db[user_id] = {
        "name": name,
        "descriptor": avg_descriptor,
        "registration_time": str(time.localtime()),
        "samples_count": len(samples)
    }

    # 保存到文件
    try:
        with open(USER_DB_PATH, "w") as f:
            json.dump(user_db, f)
        print(f"成功注册用户: {name}")
        return True
    except Exception as e:
        print(f"保存用户数据失败: {e}")
        # 从内存中删除用户
        if user_id in user_db:
            del user_db[user_id]
        return False

# 用户管理
def manage_users():
    while True:
        print("\033c")
        print("="*30)
        print("用户管理")
        print("="*30)

        if not user_db:
            print("暂无注册用户")
            print("="*30)
            print("按选择键返回")
            while not is_button_pressed(BTN_SELECT):
                time.sleep_ms(100)
            return

        # 显示用户列表
        user_list = list(user_db.items())
        for idx, (user_id, user_data) in enumerate(user_list, 1):
            print(f"{idx}. {user_data['name']} ({user_id})")

        print(f"{len(user_list)+1}. 返回")
        print("="*30)
        print("上/下: 选择 | 选择: 查看详情 | 返回: 删除用户")

        # 选择用户
        selected = 0
        while True:
            if is_button_pressed(BTN_UP):
                selected = (selected - 1) % (len(user_list) + 1)
            elif is_button_pressed(BTN_DOWN):
                selected = (selected + 1) % (len(user_list) + 1)
            elif is_button_pressed(BTN_SELECT):
                if selected < len(user_list):
                    # 查看用户详情
                    view_user_details(user_list[selected][0])
                else:
                    return  # 返回
                break
            elif is_button_pressed(BTN_BACK):
                if selected < len(user_list):
                    # 删除用户
                    delete_user(user_list[selected][0])
                break

            # 更新显示
            print("\033c")
            print("="*30)
            print("用户管理")
            print("="*30)
            for idx, (user_id, user_data) in enumerate(user_list, 1):
                prefix = "→ " if idx-1 == selected else "   "
                print(f"{prefix}{idx}. {user_data['name']} ({user_id})")
            prefix = "→ " if selected == len(user_list) else "   "
            print(f"{prefix}{len(user_list)+1}. 返回")
            print("="*30)
            print("上/下: 选择 | 选择: 查看详情 | 返回: 删除用户")

            time.sleep_ms(100)

# 查看用户详情
def view_user_details(user_id):
    user_data = user_db.get(user_id)
    if not user_data:
        print("用户不存在")
        time.sleep_ms(1000)
        return

    print("\033c")
    print("="*30)
    print(f"用户详情: {user_data['name']}")
    print("="*30)
    print(f"用户ID: {user_id}")
    print(f"注册时间: {user_data['registration_time']}")
    print(f"样本数量: {user_data['samples_count']}")
    print("="*30)
    print("按选择键返回")

    while not is_button_pressed(BTN_SELECT):
        time.sleep_ms(100)

# 删除用户
def delete_user(user_id):
    user_data = user_db.get(user_id)
    if not user_data:
        print("用户不存在")
        time.sleep_ms(1000)
        return

    print("\033c")
    print("="*30)
    print(f"确认删除用户: {user_data['name']}?")
    print("="*30)
    print("上: 确认 | 返回: 取消")

    while True:
        if is_button_pressed(BTN_UP):
            # 删除用户
            del user_db[user_id]
            try:
                with open(USER_DB_PATH, "w") as f:
                    json.dump(user_db, f)
                print(f"已删除用户: {user_data['name']}")
            except Exception as e:
                print(f"删除用户失败: {e}")
            time.sleep_ms(1000)
            break
        elif is_button_pressed(BTN_BACK):
            print("已取消删除")
            time.sleep_ms(1000)
            break
        time.sleep_ms(100)

# 主菜单
def main_menu():
    options = ["开始识别", "注册新人脸", "管理用户", "退出"]
    choice = button_menu("OpenMV人脸识别系统", options)
    return choice

# 主循环
def main():
    print("系统初始化完成")
    while(True):
        choice = main_menu()

        if choice == 0:  # 开始识别
            print("\033c")
            print("开始人脸识别，按选择键退出")
            print("="*30)

            while(True):
                img = sensor.snapshot()
                faces = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

                if faces:
                    face = faces[0]
                    img.draw_rectangle(face)

                    descriptor = extract_face_descriptor(img, face)
                    face_id = recognize_face(descriptor)

                    if face_id:
                        user_data = user_db.get(face_id)
                        if user_data:
                            img.draw_string(face[0], face[1]-15, f"{user_data['name']}", color=(0, 255, 0))
                            print(f"识别结果: {user_data['name']}")
                    else:
                        img.draw_string(face[0], face[1]-15, "未知人脸", color=(255, 0, 0))
                        print("识别结果: 未知人脸")

                if is_button_pressed(BTN_SELECT):
                    print("已退出识别模式")
                    time.sleep_ms(500)
                    break

                time.sleep_ms(100)

        elif choice == 1:  # 注册新人脸
            registration_mode()

        elif choice == 2:  # 管理用户
            manage_users()

        elif choice == 3:  # 退出
            print("系统已关闭")
            break

if __name__ == "__main__":
    main()
