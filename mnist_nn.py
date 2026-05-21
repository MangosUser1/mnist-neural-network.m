import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =============================
# 1. Загрузка и подготовка данных
# =============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Нормализация
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# =============================
# 2. Определение модели нейросети
# =============================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Первый слой
        self.fc2 = nn.Linear(128, 64)    # Второй слой
        self.fc3 = nn.Linear(64, 10)     # Выходной слой (10 классов: числа от 0 до 9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Разворачиваем размеры изображения 28x28 в вектор
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Выход без активации используется внутри loss-функции
        return x

model = SimpleNN()

# =============================
# 3. Настройка обучения
# =============================
criterion = nn.CrossEntropyLoss()  # Функция потерь
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Оптимизатор

# =============================
# 4. Обучение модели
# =============================
num_epochs = 5  # Сколько раз пройтись по всему набору данных

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Очистка предыдущих градиентов
        outputs = model(images)  # Прямой проход
        loss = criterion(outputs, labels)  # Вычисление потерь
        loss.backward()  # Обратное распространение ошибок
        optimizer.step()  # Обновление весов
        total_loss += loss.item()

    print(f"Эпоха {epoch+1}/{num_epochs}, Потери: {total_loss:.4f}")

# =============================
# 5. Тестирование модели
# =============================
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Отключаем градиенты для теста
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Находим класс с максимальным значением
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Точность на тестовых данных: {100 * correct / total:.2f}%")

# =============================
# 6. Визуализация примеров
# =============================
examples = iter(test_loader)
example_data, example_targets = next(examples)

with torch.no_grad():
    predictions = model(example_data)

# Показ первых 10 изображений и их предсказаний
for i in range(10):
    plt.imshow(example_data[i][0], cmap='gray')
    plt.title(f"Предсказание: {predictions[i].argmax().item()}")
    plt.show()