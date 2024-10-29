def flatten[T](tuple_list: tuple[list[T]]) -> list[T]:
    return [item for sublist in tuple_list for item in sublist]