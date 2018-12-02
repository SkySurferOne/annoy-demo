import random

from annoy import AnnoyIndex

f = 40
t = AnnoyIndex(f)  # tworzenie nowego indeksu, króry czyta/zapisuje f wymiarowe wektory, dom. metryka to angular
for i in range(1000):  # dodanie 1000 losowych wektorów f-wym
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10)  # zbudowanie lasu z 10 drzewami binarymi
t.save('test.ann')  # zapisanie indeksu do pliku

# ...

u = AnnoyIndex(f)
u.load('test.ann')  # wczytanie indeksu z pliku do pamięci (mmap-em), domyślnie nie wczytuje całego pliku (lazy)
res = u.get_nns_by_item(0, 1000)
print(u.get_nns_by_item(0, 1000))  # zwraca 1000 najbliższych sąsadów
