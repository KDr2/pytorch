import functools
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    overload,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import optree

T = TypeVar("T")
S = TypeVar("S")
R = TypeVar("R")

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_unflatten` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

Context = Any
PyTree = Any
TreeSpec = PyTreeSpec = optree.PyTreeSpec
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]


def _register_pytree_node(
    cls: Type,
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    namespace="torch",
) -> None:
    optree.register_pytree_node(cls, flatten_fn, unflatten_fn, namespace=namespace)


def tree_flatten(
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Tuple[List[Any], PyTreeSpec]:
    return optree.tree_flatten(pytree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_unflatten(values: Iterable[Any], spec: PyTreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, PyTreeSpec):
        raise ValueError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"PyTreeSpec but got item of type {type(spec)}."
        )
    if len(values) != spec.num_leaves:
        raise ValueError(
            f"tree_unflatten(values, spec): `values` has length {len(values)} "
            f"but the spec refers to a pytree that holds {spec.num_leaves} "
            f"items ({spec})."
        )
    return optree.tree_unflatten(spec, values)


def tree_leaves(
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> List[Any]:
    return optree.tree_leaves(pytree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_structure(
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTreeSpec:
    return optree.tree_structure(pytree, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_map(
    fn: Any,
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map(
        fn, pytree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
    )


def tree_map_(
    fn: Any,
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return optree.tree_map_(
        fn, pytree, *rests, none_is_leaf=none_is_leaf, namespace=namespace
    )


Type2 = Tuple[Type[T], Type[S]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn2 = Callable[[Union[T, S]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]

# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(ty: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(ty: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


# This specialization is needed for the implementations below that call
@overload
def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """

    def deco(f: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(f)
        def inner(x: T) -> Any:
            if isinstance(x, ty):
                return f(x)
            else:
                return x

        return inner

    return deco


@overload
def tree_map_only(
    ty: Type[T],
    fn: Fn[T, Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only(
    ty: Type2[T, S],
    fn: Fn2[T, S, Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only(
    ty: TypeAny,
    fn: FnAny[Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map(
        map_only(ty)(fn),
        pytree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@overload
def tree_map_only_(
    ty: Type[T],
    fn: Fn[T, Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


@overload
def tree_map_only_(
    ty: Type2[T, S],
    fn: Fn2[T, S, Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    ...


def tree_map_only_(
    ty: TypeAny,
    fn: FnAny[Any],
    pytree: PyTree,
    *rests: PyTree,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> PyTree:
    return tree_map_(
        map_only(ty)(fn),
        pytree,
        *rests,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


def tree_all(
    pred: Callable[[Any], bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(pytree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(pytree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    ty: Type[T],
    pred: Fn[T, bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_all_only(
    ty: Type2[T, S],
    pred: Fn2[T, S, bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_all_only(
    ty: TypeAny,
    pred: FnAny[bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(pytree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(pred(x) for x in flat_args if isinstance(x, ty))


@overload
def tree_any_only(
    ty: Type[T],
    pred: Fn[T, bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


@overload
def tree_any_only(
    ty: Type2[T, S],
    pred: Fn2[T, S, bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    ...


def tree_any_only(
    ty: TypeAny,
    pred: FnAny[bool],
    pytree: PyTree,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> bool:
    flat_args = tree_leaves(pytree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(pred(x) for x in flat_args if isinstance(x, ty))


# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(
    pytree: PyTree,
    spec: PyTreeSpec,
    *,
    none_is_leaf: bool = True,
    namespace: str = "torch",
) -> Optional[List[Any]]:
    assert isinstance(spec, PyTreeSpec)
    full_tree = tree_unflatten([0] * spec.num_leaves, spec)
    try:
        return optree.ops.broadcast_prefix(
            pytree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace
        )
    except ValueError:
        return None
