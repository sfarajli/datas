#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper for 1D indexing */
static inline int IDX(int N, int x, int y) {
	return x * N + y;
}

void print_heap_matrix(int N, int *matrix) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%3d ", matrix[IDX(N, i, j)]);
		printf("\n");
	}
}

void print_stack_matrix(int N, int matrix[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%3d ", matrix[i][j]);
		printf("\n");
	}
}

void spiral_fill_stack(int N, int matrix[N][N]) {
	int number = 1;
	int repetition = 0;

	int x = (N - 1) / 2;
	int y = (N - 1) / 2;

	matrix[x][y] = number;

	int dx = 0, dy = 1; /* initial direction: right */
	int done = 0;

	while (!done) {
		repetition++;

		for (int turn = 0; turn < 2; turn++) {
			for (int step = 0; step < repetition; step++) {
				number++;
				if (number > N * N) {
					done = 1;
					break;
				}
				x += dx;
				y += dy;
				matrix[x][y] = number;
			}

			/* Clockwise turn */
			int tmp = dx;
			dx = dy;
			dy = -tmp;
		}
	}
}

int *spiral_alloc_heap(int N) {
	if (N <= 0) return NULL;

	int *matrix = calloc((size_t)N * (size_t)N, sizeof(int));
	if (!matrix) return NULL;

	int number = 1;
	int repetition = 0;

	int x = (N - 1) / 2;
	int y = (N - 1) / 2;

	matrix[IDX(N, x, y)] = number;

	int dx = 0, dy = 1;
	int done = 0;

	while (!done) {
		repetition++;

		for (int turn = 0; turn < 2; turn++) {
			for (int step = 0; step < repetition; step++) {
				number++;
				if (number > N * N) {
					done = 1;
					break;
				}
				x += dx;
				y += dy;
				matrix[IDX(N, x, y)] = number;
			}

			int tmp = dx;
			dx = dy;
			dy = -tmp;
		}
	}

	return matrix;
}

int main(int argc, char **argv) {
	int N = 0;
	char *memory_allocation = NULL;

	if (argc == 3) {
		memory_allocation = argv[1];
		N = (int)strtol(argv[2], NULL, 10);
	} else {
		fprintf(stderr, "usage: %s [stack|heap] [number]\n", argv[0]);
		return 1;
	}

	if (N <= 0) {
		fprintf(stderr, "N must be > 0\n");
		return 1;
	}

	if (strcmp(memory_allocation, "stack") == 0) {
		int stack_matrix[N][N];           /* now safe because N>0 */
		spiral_fill_stack(N, stack_matrix);
		print_stack_matrix(N, stack_matrix);

	} else if (strcmp(memory_allocation, "heap") == 0) {
		int *heap_matrix = spiral_alloc_heap(N);

		if (!heap_matrix) {
			fprintf(stderr, "Heap allocation failed\n");
			return 1;
		}

		print_heap_matrix(N, heap_matrix);
		free(heap_matrix);

	} else {
		fprintf(stderr, "usage: %s [stack|heap] [number]\n", argv[0]);
		return 1;
	}

	return 0;
}
