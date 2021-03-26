Python-Pandas 如何shuffle（打乱）数据？
https://blog.csdn.net/qq_22238533/article/details/70917102
实现方法：
实现方法：

最简单的方法就是采用pandas中自带的 sample这个方法。

假设df是这个DataFrame

`
df.sample(frac=1)
`

这样对可以对df进行shuffle。其中参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3。
有时候，我们可能需要打混后数据集的index（索引）还是按照正常的排序。我们只需要这样操作

`
df.sample(frac=1).reset_index(drop=True)
`
-------------------------------------------------------
其实，sklearn(机器学习的库）中也有shuffle的方法。

`
from sklearn.utils import shuffle
df = shuffle(df)
`

另外，numpy库中也有进行shuffle的方法（不建议）

`
df.iloc[np.random.permutation(len(df))]
`
