#include <linux/uinput.h>
#include <unistd.h>
#include <sys/types.h>
#include "linux_virtual_device.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

VirtualDevice::VirtualDevice(DEVICE_TYPE dt)
{
    this->dt = dt;
    struct uinput_setup usetup;

    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0)
    {
        printf("[ERROR] Open /dev/uinput failed\n");
        throw "Open /dev/uinput failed!";
        return;
    }
    if (dt == VIRTUAL_MOUSE)
    {
        /*
         * The ioctls below will enable the device that is about to be
         * created, enable mouse button left and relative events.
         */
        ioctl(fd, UI_SET_EVBIT, EV_KEY);    // add key event
        ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);    // add key value
        /* EV_ABS won't work for mouse!!! */
//        ioctl(fd, UI_SET_EVBIT, EV_REL);    // add coordinate event
//        ioctl(fd, UI_SET_RELBIT, REL_X);    // add X
//        ioctl(fd, UI_SET_RELBIT, REL_Y);    // add Y
    }
    else if (dt == VIRTUAL_KEYBOARD)
    {
        ioctl(fd, UI_SET_EVBIT, EV_KEY);    // add key event
        ioctl(fd, UI_SET_KEYBIT, KEY_UP);    // add key value
        ioctl(fd, UI_SET_KEYBIT, KEY_DOWN);
        ioctl(fd, UI_SET_KEYBIT, KEY_LEFT);
        ioctl(fd, UI_SET_KEYBIT, KEY_RIGHT);
        ioctl(fd, UI_SET_KEYBIT, KEY_ENTER);
        ioctl(fd, UI_SET_KEYBIT, KEY_ESC);
    }
    else if (dt == VIRTUAL_MOUSE_KEYBOARD)
    {
        ioctl(fd, UI_SET_EVBIT, EV_KEY);    // add key event

        ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);    // add mouse key value

        ioctl(fd, UI_SET_KEYBIT, KEY_UP);    // add keyboard key value
        ioctl(fd, UI_SET_KEYBIT, KEY_DOWN);
        ioctl(fd, UI_SET_KEYBIT, KEY_LEFT);
        ioctl(fd, UI_SET_KEYBIT, KEY_RIGHT);
        ioctl(fd, UI_SET_KEYBIT, KEY_ENTER);
        ioctl(fd, UI_SET_KEYBIT, KEY_ESC);
    }
    memset(&usetup, 0, sizeof(usetup));
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor = 0x1234; /* sample vendor */
    usetup.id.product = 0x5678; /* sample product */
    strcpy(usetup.name, "gesture_virtual_device");

    ioctl(fd, UI_DEV_SETUP, &usetup);
    ioctl(fd, UI_DEV_CREATE);

    printf("[INFO] Uinput setup successfully\n");
}

VirtualDevice::~VirtualDevice()
{
    ioctl(fd, UI_DEV_DESTROY);
    close(fd);
    printf("[INFO] Uinput destroyed\n");
}

/* send event to kernel
 * Linux mouse device example:
 *       emit(fd, EV_REL, REL_X, 5); // 5 per axis
 *       emit(fd, EV_REL, REL_Y, 5);
 *       emit(fd, EV_SYN, SYN_REPORT, 0);
 */
int VirtualDevice::emit_to_kernel(int type, int code, int val)
{
    struct input_event ie;
    int ret = 0;
    ie.type = type;
    ie.code = code;
    ie.value = val;
    /* timestamp values below are ignored */
    ie.time.tv_sec = 0;
    ie.time.tv_usec = 0;

    ret = write(fd, &ie, sizeof(ie));

    return ret;
}

int VirtualDevice::emit_mouse_rel_event(int x, int y)
{
    if (emit_to_kernel(EV_REL, REL_X, x) < 0 ||
        emit_to_kernel(EV_REL, REL_Y, y) < 0 ||
        emit_to_kernel(EV_SYN, SYN_REPORT, 0) < 0)
    {
        printf("[ERROR] emit_mouse_event failed\n");
        return -1;
    }
    printf("[INFO] VIRTUAL DEVICE: emit_mouse_rel_event successfully code: (%d, %d)\n", x, y);
    return 0;
}

int VirtualDevice::emit_btn_left_event(int val)
{
    /* press:val=1 release:val=0 */
    if (emit_to_kernel(EV_KEY, BTN_LEFT, val) < 0 ||
        emit_to_kernel(EV_SYN, SYN_REPORT, 0) < 0)
    {
        printf("[ERROR] emit_btn_left_event failed\n");
        return -1;
    }
    printf("[INFO] VIRTUAL DEVICE: emit_btn_left_event successfully code: %d\n", val);
    return 0;
}

int VirtualDevice::emit_keyboard_event(KEYBOARD_KEY_TYPE key)
{
    int ret = 0;
    switch (key) {
    case VIRTUAL_KEY_UP:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_UP, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_UP, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    case VIRTUAL_KEY_DOWN:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_DOWN, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_DOWN, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    case VIRTUAL_KEY_LEFT:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_LEFT, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_LEFT, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    case VIRTUAL_KEY_RIGHT:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_RIGHT, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_RIGHT, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    case VIRTUAL_KEY_ENTER:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_ENTER, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_ENTER, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    case VIRTUAL_KEY_ESC:
        /* Key press, report the event, send key release, and report again */
        ret = emit_to_kernel(EV_KEY, KEY_ESC, 1);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        ret |= emit_to_kernel(EV_KEY, KEY_ESC, 0);
        ret |= emit_to_kernel(EV_SYN, SYN_REPORT, 0);
        break;
    }
    if (ret < 0)
        printf("[ERROR] emit_keyboard_event failed\n");
    printf("[INFO] VIRTUAL DEVICE: emit_keyboard_event successfully  code:%d\n", key);
    return ret;
}
