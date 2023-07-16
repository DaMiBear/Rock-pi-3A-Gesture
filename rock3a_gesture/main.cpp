#include "widget.h"

#include <QApplication>
#include <QFile>

#include "yolo-fastestv2-anchorfree.h"
#include "fist_track.h"
#include <string>
#include <time.h>
#include <queue>
#include <algorithm>
#include "pthread.h"
#include "cpu.h"
#include "benchmark.h"

Widget *w;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    /* load style */
    QFile file(":/style.qss");
    if (file.exists())
    {
        file.open(QFile::ReadOnly);
        QString style_sheet = QLatin1String(file.readAll());
        qApp->setStyleSheet(style_sheet);
        file.close();
    }
    w = new Widget();
    w->setWindowTitle("Gesture APP");
    w->show();
    return a.exec();
}
