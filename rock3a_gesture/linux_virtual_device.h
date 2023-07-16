#ifndef LINUX_VIRTUAL_DEVICE_H
#define LINUX_VIRTUAL_DEVICE_H

class VirtualDevice{

public:
    enum DEVICE_TYPE{
        VIRTUAL_NONE = 0,
        VIRTUAL_MOUSE,
        VIRTUAL_KEYBOARD,
        VIRTUAL_MOUSE_KEYBOARD,
    };
    enum KEYBOARD_KEY_TYPE{
        VIRTUAL_KEY_UP = 0,
        VIRTUAL_KEY_DOWN,
        VIRTUAL_KEY_LEFT,
        VIRTUAL_KEY_RIGHT,
        VIRTUAL_KEY_ENTER,
        VIRTUAL_KEY_ESC,
    };

    /* select virtual device type and create it */
    VirtualDevice(DEVICE_TYPE dt);

    /* destory the virtual device and delete it */
    ~VirtualDevice();

    /* send virtual mouse relative movement distance */
    int emit_mouse_rel_event(int x, int y);

    /* send mouse's left button val=1 means press, 0 means release */
    int emit_btn_left_event(int val);

    /* send virtual keyboard key: press then release */
    int emit_keyboard_event(KEYBOARD_KEY_TYPE key);


private:
    int fd;
    DEVICE_TYPE dt;

    /* write to the '/dev/uinput' file  */
    int emit_to_kernel(int type, int code, int val);
};

#endif // LINUX_VIRTUAL_DEVICE_H
