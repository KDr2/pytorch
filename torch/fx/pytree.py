from typing import Any, Callable, List, Tuple, Type

from torch.utils._pytree import (  # type: ignore[attr-defined]
    _register_pytree_node,
    Context,
    FlattenFunc,
    PyTree,
    PyTreeSpec,
    tree_leaves,
)

# mypy: allow-any-unimported

FlattenFuncSpec = Callable[[PyTree, PyTreeSpec], List[Any]]  # type: ignore[valid-type]


def to_flatten_func(func: FlattenFuncSpec) -> FlattenFunc:
    def flatten_func(pytree: PyTree) -> Tuple[List[Any], Context]:
        return func(pytree, None), None  # type: ignore[arg-type]

    return flatten_func


def not_implemented_unflatten(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("torch.fx.pytree: unflatten function not implemented")


def register_pytree_flatten_spec(
    typ: Type[Any], flatten_fn_spec: FlattenFuncSpec
) -> None:
    _register_pytree_node(
        typ,
        to_flatten_func(flatten_fn_spec),
        not_implemented_unflatten,
        namespace="torch.fx",
    )


def tree_flatten_spec(pytree: PyTree, spec: PyTreeSpec) -> List[Any]:  # type: ignore[valid-type]
    return tree_leaves(  # type: ignore[no-any-return]
        pytree, none_is_leaf=spec.none_is_leaf, namespace=spec.namespace
    )
