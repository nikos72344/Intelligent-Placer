# Intelligent-Placer

*Алгоритм принимает изображение горизонтальной светлой поверхности, на которой находятся:*
  - Лист формата *А4* с нарисованным маркером выпуклым многоугольником черного цвета с не более чем шестью вершинами;
  - Один или несколько обрабатываемых алгоритмом предметов.

Алгоритм, на основе полученных изображений, определяет границы и размер нарисованного многоугольника, количество предметов на изображении и их границы.
В итоге алгоритм возвращает ответ: существует ли возможность расположить обнаруженные на изображении предметы внутри распознанного многоугольника в одной плоскости.

*Формат финального ответа алгоритма:*
 - True (да), если выполнены необходимые условия работы алгоритма, и существует способ расположения распознанных предметов внутри многоугольника;
 - False (нет) - в остальных случаях.

*Объекты на изображении должны удовлетворять следующим свойствам:*
 - Объекты должны быть расположены лицевой стороной к камере;
 - Общая длина и ширина объектов не должны превышать размер расположенного рядом листа *А4*;
 - Высота каждого объекта не должна превышать *3 см*;
 - Предметы не должны перекрывать лист *А4* и друг друга на изображении;
 - Один и тот же предмет может быть использован в нескольких изображениях;
 - Каждый предмет должен обладать контрастностью в пределах *0.6 - 0.95*.

*Изображение, которое принимает алгоритм, должно удовлетворять следующим свойствам:*
 - Лист *А4* должен располагаться в верхней части изображения, а предметы – в нижней;
 - Изображение должно быть снято при искусственном источнике света, исключающем наличие бликов или теней на листе *А4* и предметах;
 - Камера при съемке должна располагаться параллельно горизонтальной поверхности, не выше *35 см* от нее;
 - Лист *А4* и предметы должны полностью попадать в кадр;
 - Разрешение изображения - *3000 x 4000*;
 - Минимальная степень размытости изображения будет дополнена.

*Набор изображений обрабатываемых алгоритмом предметов на фоне белого листа *А4*:*

![washcloth](https://user-images.githubusercontent.com/60978539/153722719-21920ae4-128f-4dae-bf36-1c374dd83f78.jpg)

![ball](https://user-images.githubusercontent.com/60978539/153722826-dfc5751d-a28b-4879-a62d-0efe37eb92cc.jpg)

![cup](https://user-images.githubusercontent.com/60978539/153722835-d13f3797-4884-48ef-ad61-c87747a75eff.jpg)

![deodorant](https://user-images.githubusercontent.com/60978539/153722860-496cbb7c-e3cc-43fc-9cc1-f1b468b9eaca.jpg)

![drug](https://user-images.githubusercontent.com/60978539/153722899-596843e6-7ce0-40f9-bbec-b3f9cedc5318.jpg)

![flash_drive](https://user-images.githubusercontent.com/60978539/153722911-ebfa73e3-760f-42f2-8995-65c7217523ee.jpg)

![pen](https://user-images.githubusercontent.com/60978539/153722921-1bddd4f3-397b-4b14-86ce-5c186f1c85ce.jpg)

![scissors](https://user-images.githubusercontent.com/60978539/153722928-3d1c416e-9f26-4c6c-86cc-0c1711d72b85.jpg)

![soap](https://user-images.githubusercontent.com/60978539/153722934-c7ed8b41-f50b-43f2-8055-8dddd63b971b.jpg)

![travel_card](https://user-images.githubusercontent.com/60978539/153722943-46bf76d8-dd5e-4a18-bd62-3a19c125b3fb.jpg)

*А также изображение горизонтальной поверхности:*

![floor](https://user-images.githubusercontent.com/60978539/153723019-ace20f38-022f-4a33-99a3-eb6d7633142b.jpg)
