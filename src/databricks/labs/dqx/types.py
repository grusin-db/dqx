from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # for type checks (mypy/vscode pylance/pyright/...) let's assume we just run native spark
    from pyspark.sql import Column
else:
    # on the real runtime, make sure that Column is covering technology we are really using
    try:
        from pyspark.sql.connect.column import Column
    except Exception:
        from pyspark.sql import Column
