import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import time

#сортировка вставками
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

#сортировка пузырьком        
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                
#сортировка выбором
def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        m = i
        for j in range(i+1, n):
            if arr[j] < arr[m]:
                m = j
        arr[i], arr[m] = arr[m], arr[i]

#сортировка слиянием        
def merge_sort(arr): 
    if len(arr) > 1: 
        mid = len(arr)//2
        left = arr[:mid] 
        right = arr[mid:]
        merge_sort(left) 
        merge_sort(right) 
        i = j = k = 0
        while i < len(left) and j < len(right): 
            if left[i] < right[j]: 
                arr[k] = left[i] 
                i+=1
            else: 
                arr[k] = right[j] 
                j+=1
            k+=1
        while i < len(left): 
            arr[k] = left[i] 
            i+=1
            k+=1
        while j < len(right): 
            arr[k] = right[j] 
            j+=1
            k+=1

#сортировка Шеллом
def shell_sort(arr): #по последовательности Шелла
    n = len(arr)
    gap = n//2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j-gap] > temp:
                arr[j] = arr[j-gap]
                j -= gap
            arr[j] = temp
        gap //= 2

def shell_sortH(arr): #по последовательности Хиббарда
    n = len(arr)
    k = int(np.log2(n))
    gap = 2**k -1
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        k -= 1
        gap = 2**k -1
    return arr

def shell_sortP(arr): #по последовательности Пратта
    # Генерация последовательности Пратта
    products = []
    pow2 = 1
    while pow2 <= len(arr):
        pow3 = 1
        while pow2 * pow3 <= len(arr):
            products.append(pow2 * pow3)
            pow3 *= 3
        pow2 *= 2
    products.sort(reverse=True)
    # Сортировка методом Шелла
    for gap in products:
        for i in range(gap, len(arr)):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
    return arr

#быстрая сортировка
def quick_sort(arr):
   if len(arr) <= 1:
       return arr
   else:
       q = random.choice(arr)
   l_nums = [n for n in arr if n < q]
 
   e_nums = [q] * arr.count(q)
   b_nums = [n for n in arr if n > q]
   return quick_sort(l_nums) + e_nums + quick_sort(b_nums)

def heap_temp(arr, n, i): #образование кучи
    largest = i 
    l = 2 * i + 1   
    r = 2 * i + 2  
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i]
        heap_temp(arr, n, largest)

#сортировка пирамидкой
def heap_sort(arr):
    n = len(arr)
    for i in range(n, -1, -1):
        heap_temp(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heap_temp(arr, i, 0)

#рисование регресионной кривой
def regress_draw(sizes,times,n,case):
    #случай для отсортированного массива
    if case==1:
        #n - номер сортировки, записано в словарь
        #0:"вставками", 1:"пузырьком", 2:"выбором", 3:"слиянием", 4:"Шелла",
        #5:"Шелла по Хиббарду", 6:"Шелла по Пратту", 7:"QuickSort", 8:"HeapSort"
        if n==0:
            x = np.array(sizes).reshape(-1,1)
            y = np.array(times)
            model = LinearRegression()
            model.fit(x,y)
            predict = model.predict(x)
        elif n==1 or n==2:
            x = np.array(sizes)
            y = np.array(times)
            X2 = x**2
            model = LinearRegression()
            model.fit(X2.reshape(-1,1),times)
            predict = model.predict(X2.reshape(-1,1))
        elif n==3 or n==7 or n==8 or n==6:
            x = np.array(sizes)
            y = np.array(times)
            log_n = x * np.log(x)
            model = LinearRegression()
            model.fit(log_n.reshape(-1,1), times)
            predict = model.predict(log_n.reshape(-1,1))
        elif n==4:
            x = np.array(sizes)
            y = np.array(times)
            log2_n = x * (np.log(x)**2)
            model = LinearRegression()
            model.fit(log2_n.reshape(-1,1), times)
            predict = model.predict(log2_n.reshape(-1,1))
        else:
            x = np.array(sizes).reshape(-1,1)**(3/2)
            y = np.array(times)
            model = LinearRegression()
            model.fit(x,y)
            predict = model.predict(x)
    #случай для случайного массива
    elif case==2 or case==4:
        if n==0 or n==1 or n==2:
            x = np.array(sizes)
            y = np.array(times)
            X2 = x**2
            model = LinearRegression()
            model.fit(X2.reshape(-1,1),times)
            predict = model.predict(X2.reshape(-1,1))
        elif n==4:
            x = np.array(sizes)
            y = np.array(times)
            X32 = x**(3/2)
            model = LinearRegression()
            model.fit(X32.reshape(-1,1),times)
            predict = model.predict(X32.reshape(-1,1))
        elif n==3 or n==7 or n==8:
            x = np.array(sizes)
            y = np.array(times)
            log_n = x * np.log(x)
            model = LinearRegression()
            model.fit(log_n.reshape(-1,1), times)
            predict = model.predict(log_n.reshape(-1,1))
        elif n==5:
            x = np.array(sizes).reshape(-1,1)**(5/4)
            y = np.array(times)
            model = LinearRegression()
            model.fit(x,y)
            predict = model.predict(x)
        elif n==6:
            x = np.array(sizes)
            y = np.array(times)
            log2_n = x * (np.log(x)**2)
            model = LinearRegression()
            model.fit(log2_n.reshape(-1,1), times)
            predict = model.predict(log2_n.reshape(-1,1))
    #случай для обратноотсортированного массива
    elif case==3:
        if n==0 or n==1 or n==2 or n==4 or n==7:
            x = np.array(sizes)
            y = np.array(times)
            X2 = x**2
            model = LinearRegression()
            model.fit(X2.reshape(-1,1),times)
            predict = model.predict(X2.reshape(-1,1))
        elif n==6:
            x = np.array(sizes)
            y = np.array(times)
            log2_n = x * (np.log(x)**2)
            model = LinearRegression()
            model.fit(log2_n.reshape(-1,1), times)
            predict = model.predict(log2_n.reshape(-1,1))
        elif n==5:
            x = np.array(sizes).reshape(-1,1)**(3/2)
            y = np.array(times)
            model = LinearRegression()
            model.fit(x,y)
            predict = model.predict(x)
        else:
            x = np.array(sizes)
            y = np.array(times)
            log_n = x * np.log(x)
            model = LinearRegression()
            model.fit(log_n.reshape(-1,1), times)
            predict = model.predict(log_n.reshape(-1,1))
    plt.plot(sizes, predict, color='red', label="Регресионная кривая")
    plt.legend()

#рисовка результатов
def plot_results(sizes, times, n, case):
    name = {0:"вставками", 1:"пузырьком", 2:"выбором", 3:"слиянием", 4:"Шелла", 5:"Шелла по Хиббарду", 6:"Шелла по Пратту", 7:"QuickSort", 8:"HeapSort"}
    #для удобства графики поделены на три
    if(n<3):
        plt.subplot(1,3,n+1)
        plt.scatter(sizes, times, marker='o')
        plt.xlabel('Количество элементов n')
        plt.ylabel('Время в секундах')
        plt.title(f"Сортировка {name[n]}")
        regress_draw(sizes,times,n,case)
        plt.grid()
    elif n>=3 and n<=5:
        plt.subplot(1,3,n+1-3)
        plt.scatter(sizes, times, marker='o')
        plt.xlabel('Количество элементов n')
        plt.ylabel('Время в секундах')
        plt.title(f"Сортировка {name[n]}")
        regress_draw(sizes,times,n,case)
        plt.grid()
    else:
        plt.subplot(1,3,n+1-6)
        plt.scatter(sizes, times, marker='o')
        plt.xlabel('Количество элементов n')
        plt.ylabel('Время в секундах')
        plt.title(f"Сортировка {name[n]}")
        regress_draw(sizes,times,n,case)
        plt.grid()
    if n==2 or n==5:
        #regress_draw(sizes,times,n)
        plt.show()
        print("Для продолжения работы закройте график....")

#запись времени для каждой сортировки
def time_pass(n, random_arr):
    arr=random_arr.copy()
    if n==0:
        print(f"Сортировка вставками для {len(arr)}",end=" ")
        start = time.time()
        insertion_sort(arr)
        end = time.time()
        return end-start
    elif n==1:
        print(f"Сортировка пузырьком для {len(arr)}",end=" ")
        start = time.time()
        bubble_sort(arr)
        end = time.time()
        return end-start
    elif n==2:
        print(f"Сортировка выбором для {len(arr)}",end=" ")
        start = time.time()
        selection_sort(arr)
        end = time.time()
        return end-start
    elif n==3:
        print(f"Сортировка слиянием для {len(arr)}",end=" ")
        start = time.time()
        merge_sort(arr)
        end = time.time()
        return end-start
    elif n==4:
        print(f"Сортировка Шелла для {len(arr)}",end=" ")
        start = time.time()
        shell_sort(arr)
        end = time.time()
        return end-start
    elif n==5:
        print(f"Сортировка Шелла по Хиббарду для {len(arr)}",end=" ")
        start = time.time()
        shell_sortH(arr)
        end = time.time()
        return end-start
    elif n==6:
        print(f"Сортировка Шелла по Пратту для {len(arr)}",end=" ")
        start = time.time()
        shell_sortP(arr)
        end = time.time()
        return end-start
    elif n==7:
        print(f"Сортировка Quick для {len(arr)}",end=" ")
        start = time.time()
        quick_sort(arr)
        end = time.time()
        return end-start
    elif n==8:
        print(f"Сортировка Heap для {len(arr)}",end=" ")
        start = time.time()
        heap_sort(arr)
        end = time.time()
        return end-start

def time_output(times):
    print("Ответ:")
    names = {0:"вставками", 1:"пузырьком", 2:"выбором", 3:"слиянием", 4:"Шелла", 5:"Шелла по Хиббарду", 6:"Шелла по Пратту", 7:"QuickSort", 8:"HeapSort"}
    temp = times[:8]
    min_time_index = times.index(min(temp))
    for i in range(0,9):
        print(f"В отсортированном массиве сортировка {names[i]} заняла: {temp[i]}")
    print(f"В этом случая, самой быстрой оказалась сортировка {names[min_time_index]}")
    temp = times[8:16]
    for i in range(0,9):
        print(f"В случайном массиве сортировка {names[i]} заняла: {temp[i]}")
    min_time_index = times.index(min(temp))
    print(f"В этом случая, самой быстрой оказалась сортировка {names[min_time_index]}")
    temp = times[16:24]
    for i in range(0,9):
        print(f"В обратноотсортированном массиве сортировка {names[i]} заняла: {temp[i]}")
    min_time_index = times.index(min(temp))
    print(f"В этом случая, самой быстрой оказалась сортировка {names[min_time_index]}")
    temp = times[24:32]
    for i in range(0,9):
        print(f"В почти отсортированном массиве сортировка {names[i]} заняла: {temp[i]}")
    min_time_index = times.index(min(temp))
    print(f"В этом случая, самой быстрой оказалась сортировка {names[min_time_index]}")
    

#---------main------------
array_size_n = [i for i in range(100000,1000000,100000)]
array_size_n2 = [i for i in range(4000,25000,5000)]
array_size_nlogn = [i for i in range(100000,1000000,100000)]
time_array = []
final_time = []
final_time1 = []
final_time2 = []
case = 1
print("----Отсортированный массив----")
sort_time = 0
for size in array_size_n:
    sorted_array = [i for i in range(1,size+1)]
    sort_time = time_pass(0,sorted_array)
    print(sort_time)
    time_array.append(sort_time)
plot_results(array_size_n,time_array,0,case)
final_time.append(sort_time)
time_array.clear()
for n in range(1,3):
    sort_time = 0
    for size in array_size_n2:
        sorted_array = [i for i in range(1,size+1)]
        sort_time = time_pass(n,sorted_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_n2,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
for n in range(3,9):
    sort_time = 0
    for size in array_size_nlogn:
        sorted_array = [i for i in range(1,size+1)]
        sort_time = time_pass(n,sorted_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_nlogn,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
plt.subplots_adjust(wspace=0.367, hspace=0.429)
plt.show()
print("----Случайный массив----")
case+=1
for n in range(0,3):
    sort_time = 0
    for size in array_size_n2:
        random_array = [random.randint(1, 100) for _ in range(size)]
        sort_time = time_pass(n,random_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_n2,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
for n in range(3,9):
    sort_time = 0
    for size in array_size_nlogn:
        random_array = [random.randint(1, 100) for _ in range(size)]
        sort_time = time_pass(n,random_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_nlogn,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
plt.subplots_adjust(wspace=0.367, hspace=0.429)
plt.show()
print("----Обратноотсортированный массив----")
case+=1
sort_time = 0
for n in range(0,3):
    sort_time = 0
    for size in array_size_n2:
        back_array = [i for i in range(size,0,-1)]
        sort_time = time_pass(n,back_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_n2,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
for n in range(3,9):
    sort_time = 0
    for size in array_size_nlogn:
        back_array = [i for i in range(size,0,-1)]
        sort_time = time_pass(n,back_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_nlogn,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
plt.subplots_adjust(wspace=0.367, hspace=0.429)
plt.show()
print("----Почти отсортированный массив 90/10----")
case+=1
sort_time = 0
for n in range(0,3):
    sort_time = 0
    for size in array_size_n2:
        chance = round(size*0.9)
        almost_array = [i for i in range(1,chance+1)]
        random_array = [random.randint(1,100) for _ in range((size-chance)+1)]
        almost_array+=random_array
        sort_time = time_pass(n,almost_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_n2,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
for n in range(3,9):
    sort_time = 0
    for size in array_size_nlogn:
        chance = round(size*0.9)
        almost_array = [i for i in range(1,chance+1)]
        random_array = [random.randint(1,100) for _ in range((size-chance)+1)]
        almost_array+=random_array
        sort_time = time_pass(n,almost_array)
        print(sort_time)
        time_array.append(sort_time)
    plot_results(array_size_nlogn,time_array,n,case)
    final_time.append(sort_time)
    time_array.clear()
plt.subplots_adjust(wspace=0.367, hspace=0.429)
plt.show()