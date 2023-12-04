#include <iostream>
#include <vector>
#include <algorithm>

void evenOddSort(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end(), [](int a, int b) {
        if (a % 2 == 0 && b % 2 == 0) {
            return a < b;  // Sort even numbers in ascending order
        } else if (a % 2 != 0 && b % 2 != 0) {
            return a < b;  // Sort odd numbers in ascending order
        } else {
            return a % 2 < b % 2;  // Place even numbers before odd numbers
        }
    });
}

int main() {
    std::vector<int> numbers = {5, 2, 9, 1, 5, 6};

    evenOddSort(numbers);

    std::cout << "Sorted Array: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}




