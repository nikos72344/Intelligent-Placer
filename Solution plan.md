# План решения задачи

***

***Обработка листа *А4* и многоугольника***

**Предобработка изображения:**
  1) Проверяем наличие файла на жестком диске по переданному в программу пути;
  2) Считываем изображение в серых тонах;
  3) Сглаживаем прочитанное изображение;
  4) Проводим бинаризацию;
  5) Находим контуры на преобразованном изображении.

***

**Обработка полученной информации:**
  1) К обработке допускаются контуры, центры которых располагаются в верхней половине изображения;
  2) Сравниваем распознанные контуры по периметру - предпоследний контур с наибольшим периметром будет внутренним контуром листа. Внутренний контур многоугольника будет находиться на четвертом месте с конца отсортированного по периметру списка контуров; 
  3) Находим вершины и площадь полученных контуров.

***Обработка расположенных в нижней части объектов***
  1) К обработке допускаются внешние контуры, центры которых расположены в нижней половине изображения; 
  2) Контуры сортируются в список на основании их периметра (по возрастанию);
  3) Не более десяти последних (с наибольшей площадью) контуров в списке будут рассматриваться как предметы;
  4) У каждого предмета будет происходить вычисление площади. 

**Работа с изображением будет проводиться с использованием библиотеки OpenCV.**

***

***Основной алгоритм***

Алгоритм на основании задачи раскроя будет определять возможно ли размещение распознанных предметов в нарисованном многоугольнике.
В случае нахождения возможного размещения, будет возвращен результат True, в противном случае - False.
