---
title: MathJax的基本用法
date: 2017-08-13 19:57:57
tags:
- MathJax
---
本文主要介绍MathJax的符号表示以及常用语法表示。
<!--more-->
# 插入公式
* 如果是文本中插入公式，则用 ` $ ... $ `
* 如果是单独的公式行，则使用 ` $$ ... $$ `

# 多行公式
如果需要些多行公式，就用
```MathJax
\begin{equation}\begin{split}
...
end{split}\end{equation}
```
`\\`表示换行，`&`表示要对其的位置，例如：
```MathJax
\begin{equation}\begin{split}
H(Y|X) &=\sum_{x\in X} p(x)H(Y|X)\\
&=-\sum_{x\in X} p(x)\sum_{y\in Y}p(y|x)\log p(y|x)\\
&=-\sum_{x\in X} \sum_{y\in Y}p(y,x)\log p(y|x)
\end{split}\end{equation}
```

$$
\begin{equation}\begin{split}
H(Y|X) &=\sum{x\in X} p(x)H(Y|X) \\\\
&=-\sum{x\in X} p(x)\sum{y\in Y}p(y|x)\log p(y|x) \\\\
&=-\sum{x\in X} \sum_{y\in Y}p(y,x)\log p(y|x)
\end{split}\end{equation}
$$


# 分式
有两种方法可以实现分式：
* 使用`\frac a b`。例如`\frac {1+a} {4+b}`，效果为：$\frac {1+a} {4+b}$

* 使用 `a \over b`。例如`{1+a} \over {4+b}`，效果为：${1+a} \over {4+b}$

不要在指数或者积分中使用`\frac`。在指数或者积分中使用`\frac`会使表达式看起来不清晰，因此在专业的数学排版中很少被使用。应该使用`/`来代替。
```MathJax
$$
\begin{array}
{c | c} \\
\mathrm{Bad} & \mathrm{Better} \\
\hline \\
e^{i \frac {\pi} 2} \quad e^{\frac{i \pi} 2} &
e^{i \pi / 2} \\
\int _ {- \frac \pi 2}^ \frac \pi 2 \sin x \, dx &
\int _ {- \pi / 2}^{\pi / 2}\sin x \, dx \\
\end{array}
$$
```
效果如下：

$$
\begin{array}
{c | c} \\\\
\mathrm{Bad} & \mathrm{Better} \\\\
\hline \\\\
e^{i \frac {\pi} 2} \quad e^{\frac{i \pi} 2} &
e^{i \pi / 2} \\\\
\int \_ {- \frac \pi 2}^ \frac \pi 2 \sin x \, dx &
\int _ {- \pi / 2}^{\pi / 2}\sin x \, dx \\\\
\end{array}
$$

书写连分数表达式的时候，使用`\cfrac`来代替`\frac`或`\over`。
```MathJax
$$
\begin{array}
{c | c}
\mathrm{Bad(over)} & \mathrm{Bad(frac)} & \mathrm{Better(cfrac)} \\
\hline \\
x = a_0 + { {1^2} \over {a_1 + { {2^2} \over {a_2 + { {3^2} \over {a_3 + { {4^4} \over {a_4 + \cdots}}}}}}}} &
x = a_0 + \frac {1^2}{a_1 + \frac {2^2} {a_2 + \frac {3^2} {a_3 + \frac{4^4} {a_4 + \cdots}}}} &
x = a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2} {a_2 + \cfrac {3^2} {a_3 + \cfrac{4^4} {a_4 + \cdots}}}}
\end{array}
$$
```
效果

$$
\begin{array}
{c | c}
\mathrm{Bad(over)} & \mathrm{Bad(frac)} & \mathrm{Better(cfrac)} \\\\
\hline \\\\
x = a_0 + { {1^2} \over {a_1 + { {2^2} \over {a_2 + { {3^2} \over {a_3 + { {4^4} \over {a_4 + \cdots}}}}}}}} &
x = a_0 + \frac {1^2}{a_1 + \frac {2^2} {a_2 + \frac {3^2} {a_3 + \frac{4^4} {a_4 + \cdots}}}} &
x = a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2} {a_2 + \cfrac {3^2} {a_3 + \cfrac{4^4} {a_4 + \cdots}}}}
\end{array}
$$

# 根式
* 平方根。`\sqrt {x^3}`：效果为$\sqrt {x^3}$。

* 其他根式。`\sqrt[4] {\frac x y}`：效果为$\sqrt[4] {\frac x y}$。

# 公式的对齐
有时候可能需要一系列的公式中等号对齐。需要用到`$ \begin{align} ... \end{align}$`的格式，其中需要使用`&`来指示要对齐的位置。
```MathJax
$$
\begin{align}
\sqrt{37}
&= \sqrt{\frac {73^2-1} {12^2}} \\
&= \sqrt{\frac {73^2} {12^2} \cdot \frac {73^2-1} {73^2}} \\
&= \sqrt{\frac {73^2} {12^2}} \sqrt {\frac {73^2-1} {73^2}} \\
&= \frac {73} {12} \sqrt{1 - \frac {1} {73^2}} \\
&\approx \frac {73} {12} \left( 1 - \frac {1} {2 \cdot 73^2} \right)
\end{align}
$$
```
效果如下：
$$
\begin{aligned}
\sqrt{37}
&= \sqrt{\frac {73^2-1} {12^2}} \\\\
&= \sqrt{\frac {73^2} {12^2} \cdot \frac {73^2-1} {73^2}} \\\\
&= \sqrt{\frac {73^2} {12^2}} \sqrt {\frac {73^2-1} {73^2}} \\\\
&= \frac {73} {12} \sqrt{1 - \frac {1} {73^2}} \\\\
&\approx \frac {73} {12} \left( 1 - \frac {1} {2 \cdot 73^2} \right)
\end{aligned}
$$

# 公式的标记与引用
使用`\tag{yourtag}`来标记公式，如果后文想要引用该公式，则还需要在`\tag{yourtag}`之后加上`\label{yourlabel}`，例如：
```Mathjax
$$
a = x^2 - y^3 \tag{公式1}\label{label1}
$$
```
效果为：
$$
a = x^2 - y^3 \tag{公式1}\label{label1}
$$
如果需要引用该公式，需要使用`\eqref{label}`，例如：
```Mathjax
$$
a+ y^3  \stackrel {\eqref {label1}} = x^2
$$
```
效果如下：
$$
a+ y^3  \stackrel {\eqref {label1}}  = x^2
$$
可以看到，通过超链接可以跳转到${\eqref {label1}} $的位置。



# 字体
* 使用`\mathbb`来显示黑板粗体字:

 $ \mathbb {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathbb {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathbf`来显示粗体字:

 $ \mathbf {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathbf {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathtt`来显示打印字体

 $ \mathtt {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathtt {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathrm`来显示罗马字体:

 $ \mathrm {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathrm {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathcal`来显示手写字体:

 $ \mathcal {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathcal {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathbf`来显示剧本字体:

 $ \mathbf {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathbf {abcdefghijklmnopqrstuvwxyz} $

* 使用`\mathfrak`来显示Fraktur字母（一种旧的德国字体）:

 $ \mathfrak {ABCDEFGHIGKLMNOPQRSTUVWXYZ} $
 $ \mathfrak {abcdefghijklmnopqrstuvwxyz} $

# 分组
通过大括号`{}`将操作数与符号分隔开，消除二义性。

例如：`x^10`的效果为$x^10$，如果在两个数字上加上大括号，`x^{10}`，最终效果为$x^{10}$

# 空间
MathJax通常有一套复杂的策略来决定公式的空间距离，直接在两届元素之间加入空格是毫无用处的。因此为了增加空间距离，使用`\,`可以增加少许的空间；使用`\;`可以增加更多地空间；`\quad`和`\qquad`分别对应更多地空间。
```MathJax
$ a \, b \; c \quad d \qquad e f g $
```
效果如下：
$ a \, b \; c \quad d \qquad e f g $


# 希腊字母
| 大写字母 | 代码 | 小写字母 | 代码 |
|:--------------:|:--------:|:-----------:|:---:|
| $A$ | `A` | $\alpha$ | `\alpha` |
| $B$ | `B` | $\beta$ | `\beta` |
| $\Gamma$ | `\Gamma` | $\gamma$ | `\gamma` |
| $\Delta$ | `\Delta` | $\delta$ | `\delta` |
| $E$ | `E` | $\epsilon$ | `\epsilon` |
| $Z$ | `Z` | $\zeta$ | `\zeta` |
| $H$ | `H` | $\eta$ | `\eta` |
| $\Theta$ | `\Theta` | $\theta$ | `\theta` |
| $ \Lambda $ | `\Lambda` | $\lambda$ | `\lambda` |
| $M$ | `M` | $\mu$ | `\mu` |
| $N$ | `N` | $\nu$ | `\nu` |
| $ \Xi$ | `\Xi` | $ \xi $ | ` \xi ` |
| $ O$ | `O` | $\omicron$ | `\omicron` |
| $ \Pi$ | `\Pi` | $\pi$ | `\pi` |
| $ P$ | `P` | $\rho$ | `\rho` |
| $ \Sigma$ | `\Sigma` | $\sigma$ | `\sigma` |
| $ T$ | `T` | $\tau$ | `\tau` |
| $ \Upsilon$ | `\Upsilon` | $\upsilon$ | `\upsilon` |
| $ \Phi$ | `\Phi` | $\phi$ | `\phi` |
| $ X $ | ` X ` | $ \chi $ | `\chi` |
| $ \Psi$ | `\Psi` | $\psi$ | `\psi` |
| $ \Omega$ | `\Omega` | $\omega$ | `\omega` |

# 数学符号

## 上标与下标
上标和下标只需要在后面加上`^`或`_`，需要注意的是，如果上表或者下表不止有一个字符的话需要用大括号`{}`括起来。

|运算符 | 说明 | 示例代码 | 效果 |
|:--------:| :------------:|:-----------:|:--------:|
| `^` | 上标 | `$x^y$` | $x^y$|
| `_` | 下标 | `$x_y$` | $x_y$|
|  `\mid `| 上下限 | `$\mid _a^b$ ` | $\mid _a^b$|
| `\sideset` | 四周标记| `$\sideset {^1_2} {^3_4} \bigotimes$` | $\sideset {^1_2} {^3_4} \bigotimes$|
| `choose` | 选择排列 | `$n+1 \choose 2k$` | $n+1 \choose 2k$|
| `\binom` | 二项式排列| `$\binom {n+1} {2k}$` | $\binom {n+1} {2k}$|


## 关系比较符号
| 符号 | 代码 |
| :-------:| :-------:|
| $ \lt $ | `\lt`|
| $ \gt $ | `\gt `|
| $ \le $ | `\le `|
| $\ge $ | `\ge `|
| $ \neq $ | `\neq `|
| $ \not\lt $ | `\not\lt `|
| $ \nleq $ | `\nleq `|
| $ \not\gt $ | `\not\gt `|
| $ \ngeq $ | `\ngeq `|
| $\approx$ | `\approx` |
| $\equiv$ | `\equiv` |
| $\sim$ | `\sim` |
| $\cong$ | `\cong` |
| $\prec$ | `\prec` |

## 运算符号
| 运算符| 代码 |
| :-------:| :-------:|
|  $ + $ |  ` + `  |
|  $ -$ |  ` -`  |
|  $ \times$ |  ` \times`  |
|  $ \div $ |  ` \div `  |
|  $ \pm $ |  ` pm`  |
|  $ \mp $ |  ` mp`  |
|  $ \cdot $ |  ` \cdot`  |
|  $ \ast $ |  ` \ast`  |
|  $ \pmod n$ |  ` \pmod n`  |
|  $ \mid $ |  ` \mid`  |
|  $ \nmid $ |  ` \nmid`  |
|  $ \sum $ |  ` \sum`  |
|  $ \prod $ |  ` \prod`  |
|  $ \coprod $ |  ` \coprod`  |
|  $ \oplus $ |  ` \oplus`  |
|  $ \odot $ |  ` \odot`  |
|  $ \otimes $ |  ` \otimes`  |
|  $ \bigoplus $ |  ` \bigoplus`  |
|  $ \bigodot $ |  ` \bigodot`  |
|  $ \bigotimes $ |  ` \bigotimes`  |

## 集合符号
| 运算符| 代码 |
| :-------:| :-------:|
|  $ \cup $ |  ` \cup `  |
|  $ \not\cup $ |  ` \not\cup `  |
|  $ \cap $ |  ` \cap `  |
|  $ \not\cap $ |  ` \not\cap `  |
|  $ \setminus $ |  ` \setminus `  |
|  $ \subset $ |  ` \subset `  |
|  $ \not\subset $ |  ` not\subset `  |
|  $ \subseteq $ |  ` \subseteq `  |
|  $ \not\subseteq $ |  ` not\subseteq `  |
|  $ \subsetneq $ |  ` \subsetneq `  |
|  $ \supset $ |  ` \supset `  |
|  $ \not\supset $ |  ` \not\supset `  |
|  $ \supseteq $ |  ` \supseteq `  |
|  $ \not\supseteq $ |  ` \not\supseteq `  |
|  $ \in $ |  ` \in `  |
|  $ \notin $ |  ` \notin `  |
|  $ \emptyset $ |  ` \emptyset `  |
|  $ \varnothing $ |  ` \varnothing `  |
|  $ \vee $ 和取 |  ` \vee `  |
|  $ \not\vee $ 非和取|  ` \not\vee `  |
|  $ \wedge $ 析取|  ` \wedge `  |
|  $ \not\wedge $非析取 |  ` \not\wedge `  |
|  $ \uplus $ |  ` \uplus `  |
|  $ \not\uplus $ |  ` \not\uplus `  |
|  $ \sqcup $ |  ` \sqcup `  |
|  $ \not\sqcup $ |  ` \not\sqcup `  |
|  $ \bigcup $ |  ` \bigcup `  |
|  $ \not\bigcup $ |  ` \not\bigcup `  |
|  $ \bigvee $ |  ` \bigvee `  |
|  $ \not\bigvee $ |  ` \not\bigvee `  |
|  $ \bigwedge $ |  ` \bigwedge `  |
|  $ \not\bigwedge $ |  ` \not\bigwedge `  |
|  $ \biguplus $ |  ` \biguplus `  |
|  $ \not\biguplus $ |  ` \not\biguplus `  |
|  $ \bigsqcup $ |  ` \bigsqcup `  |
|  $ \not\bigsqcup $ |  ` \not\bigsqcup `  |


## 箭头符号
| 运算符| 代码 |
| :-------:| :-------:|
|  $ \to $ |  ` \to `  |
|  $ \mapsto $ |  ` \mapsto `  |
|  $ \Rightarrow $ |  ` \Rightarrow `  |
|  $ \rightarrow $ |  ` \rightarrow `  |
|  $ \Longrightarrow $ |  ` \Longrightarrow `  |
|  $ \longrightarrow $ |  ` \longrightarrow `  |
|  $ \Leftarrow $ |  ` \Leftarrow `  |
|  $ \leftarrow $ |  ` \leftarrow `  |
|  $ \Uparrow $ |  ` \Uparrow `  |
|  $ \uparrow $ |  ` \uparrow `  |
|  $ \Downarrow $ |  ` \Downarrow `  |
|  $ \downarrow $ |  ` \downarrow `  |
|  $ \dagger $（剑标）|  ` \dagger `  |
|  $ \ddagger $（双剑标）|  ` \ddagger `  |

## 特殊符号
| 运算符| 代码 |
| :-------:| :-------:|
|  $ \infty $ |  ` \infity `  |
|  $ \nabla $ |  ` \nabla `  |
|  $ \partial $ |  ` \partial `  |
|  $ \approx $ |  ` \approx `  |
|  $ \sim $ |  ` \sim `  |
|  $ \simeq $ |  ` \simeq `  |
|  $ \cong $ |  ` \cong `  |
|  $ \equiv $ |  ` \equiv `  |
|  $ \prec $ |  ` \prec `  |
|  $ {n+1 \choose 2k } $ |  ` {n+1 \choose 2k }` 或 `  \binom {n+1} {2k} `  |
|  $ \land $ |  ` \land `  |
|  $ \lor $ |  ` \lor `  |
|  $ \lnot $ |  ` \lnot `  |
|  $ \forall $ |  ` \forall `  |
|  $ \exists $ |  ` \exists `  |
|  $ \top $ |  ` \top `  |
|  $ \bot $ |  ` \bot `  |
|  $ \vdash $ |  ` \vdash `  |
|  $ \vDash $ |  ` \vDash ` |
|  $ \star $ |  ` \star ` |
|  $ \ast $ |  ` \ast ` |
|  $ \oplus $ |  ` \oplus ` |
|  $ \circ $ |  ` \circ ` |
|  $ \bullet $ |  ` \bullet ` |

## 括号
需要注意的是，原始的符号不会随着公式的大小自动缩放，可以使用`\left`、`\right`来自适应调整括号$()$、$[]$、$ \{\}$以及分隔符$|$的大小。如果需要省略部分括号内容，可以用`\left.`或`\right.`来代替。
```MathJax
$$
\begin{aligned}
( \frac 1 2 ) &= [\frac 1 2] \\
\left( \frac 1 2 \right) &= \left[ \frac 1 2 \right] \\
\lbrace \sum _{i=0}^n i^2 \rbrace &= \langle \frac {( \frac {n}{2} + n)(2n+1)}{6} \rangle \\
\left \lbrace \sum _{i=0}^n i^2 \right\rbrace &= \left\langle \frac {\left( \frac {n}{2} + n \right)(2n+1)}{6} \right \rangle \\
\left. \sum _{i=0}^n i^2 \right\rbrace &= \left\langle \frac {\left( \frac {n}{2} + n \right)(2n+1)}{6} \right. \\
\left. \frac {d u} {d x} \right| _{x=0} &= 1
\end{aligned}
$$
```
效果如下：

$$
\begin{aligned} 
( \frac 1 2 ) &= [\frac 1 2] \\\\
\left( \frac 1 2 \right) &= \left[ \frac 1 2 \right] \\\\
\lbrace \sum \_{i=0}^n i^2 \rbrace &= \langle \frac {( \frac {n}{2} + n)(2n+1)}{6} \rangle \\\\
\left \lbrace \sum \_{i=0}^n i^2 \right\rbrace &= \left\langle \frac {\left( \frac {n}{2} + n \right)(2n+1)}{6} \right \rangle \\\\
\left . \sum \_{i=0}^n i^2 \right\rbrace &= \left\langle \frac {\left( \frac {n}{2} + n \right)(2n+1)}{6} \right . \\\\
\left. \frac {d u} {d x} \right| \_{x=0} &= 1
\end{aligned}
$$

| 运算符| 说明 | 代码 |
| :-------:| :-----:|:-------:|
|  $ () $ |小括号|  `()`  |
|  $ [] $ |中括号|  `[] `  |
|  $\\{ \\} $ | 大括号 |  `\{ \} ` 或`\lbrace \rbrace` |
|  $ \langle \rangle  $ |尖括号|  ` \langle \rangle` |
|  $ \lceil x \rceil $  | 上取整|  ` \lceil x \rceil ` |
|  $ \lfloor x \rfloor $  | 下取整|  ` \lfloor x \rfloor` |


## 对数运算
| 运算符  |  示例代码 | 效果  |
| :------------: | :------------: | :------------: |
|  `\log` |  `$\log(x)$` | $\log(x)$  |
|  `\lg` |  `$\lg(x)$` |  $\lg(x)$ |
|  `\ln` |  `$\ln(x)$` |  $\ln(x)$ |


## 顶部符号与连线符号
| 运算符| 代码 |
| :-------:| :-------:|
|  $\hat x $ |  ` \hat x `  |
|  $\widehat  {xy} $ |  ` \widehat {xy} `|
|  $\overline {xyz} $ |  ` \overline {xyz} `|
|  $\vec {ab} $ |  ` \vec {ab} `|
|  $\overrightarrow {abcd} $ |  ` \overrightarrow  {abcd} `|
|  $ \dot d $ |  ` \dot d `|
|  $ \ddot d $ |  ` \ddot d `|
| $ \tilde a $ | ` \tilde a ` |

## 三角运算符
|  运算符 | 说明  | 示例代码  | 效果  |
| :------------: | :------------: | :------------: | :------------: |
|  `\bot` |  垂直 |  `$A \bot B$` | $A \bot B$  |
|  `\angle` | 角  |  `$\angle 45$` | $\angle 45$  |
|  `circ` |  度 |  `$45^\circ$` |  $45^\circ$ |
|  `\sin` |  正弦 |  `$\sin 30^\circ = 0.5$` | $\sin 30^\circ = 0.5$  |
|  `\cos` |  余弦 |  `$\cos 90^\circ = 0$` | $\cos 90^\circ = 0$  |
|  `\tan` |  正切 |  `$\tan 45^\circ = 1$` | $\tan 45^\circ = 1$  |
|  `\arcsin` |  反正弦 | `$\arcsin 0.5 = 30^\circ$`  |  $\arcsin 0.5 = 30^\circ$ |
|  `\arccos` |  反余弦 | `$\arcsin 0.5 = 60^\circ$`  |  $\arcsin 0.5 = 60^\circ$ |
|  `\arctan` |  反正切 | `$\arcsin 0.5 = 45^\circ$`  |  $\arcsin 0.5 = 45^\circ$ |
|  `\cot` |  余切 | `$\cot$`  | $\cot$  |
|  `\sec` |  正割 | `$\sec$`  | $\sec$  |
|  `\csc` |  余割 | `$\csc$`  | $\csc$  |

## 微积分运算符
| 运算符  |  效果 |
| :------------: | :------------: |
|  `\prime` | $\prime$  |
|  `\int` | $\int$  |
|  `\iint` |  $\iint$ |
|  `\iiint` | $\iiint$  |
|  `\iiiint` | $\iiiint$  |
|  `\oint` | $\oint$  |
|  `\lim` |  $\lim$ |
|  `\infty` | $\infty$  |
|  `\nabla` |  $\nabla$ |
|  `\partial` |  $\partial$ |
块公式显示`$\displaystyle \lim_{x\to\infty}$`:$\displaystyle \lim_{x\to\infty}$

## 逻辑运算符
| 运算符  | 效果  |
| :------------: | :------------: |
|  `\because` |  $\because$ |
|  `\therefore` | $\therefore$  |
|  `\land` |  $\land$ |
|  `\lor` |  $\lor$ |
|  `\lnot` |  $\lnot$ |
|  `\forall` | $\forall$  |
|  `\exists` | $\exists$  |
|  `\top` |  $\top$ |
|  `\bot` |  $\bot$ |
|  `\vdash` | $\vdash$  |
|  `\vDash` | $\vDash$  |


## 其他符号
|  运算符 | 效果  |
| :------------: | :------------: |
|  `\ldots`底端对齐的省略号 |  $\ldots$ |
|  `\cdots`中线对齐的省略号 | $\cdots$  |
|  `\vdots`竖直对齐的省略号 | $\vdots$  |
|  `\ddots`矩阵对齐的省略号 | $\ddots$  |
|  `\star` | $\star$  |
|  `\ast` |  $\ast$ |
|  `\cirs` |  $\circ$ |
|  `\bullet` | $\bullet$  |
|  `\bigstar` |  $\bigstar$ |
|  `\bigcirc` | $\bigcirc$  |
|  `\aleph` | $\aleph$  |
|  `\Im` |  $\Im$ |
|  `\Re` | $\Re$  |


# 表格
在MathJax中插入表格需要 `$$  \begin{array} {列格式} ... \end{array} $$`，在`\begin{array}`后面需要表明每一列的格式：`c`表示居中；`l`表示左对齐，`r`表示右对齐；`|`表示列分割线；`\\`表示每一行的结束；`&`用来分割矩阵元素；`\hline`表示行分割线；使用`\text{文字内容}`在表格中插入文本。`%`来添加注释。
```MathJax\
$$
\begin{array}{c | lcr}
n &  \text{Left} &  \text{Center} &   \text{Right} \\  
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i
\end{array}
$$
```
效果如下：
$$
\begin{array}{c | lcr}
n &  \text{Left} &  \text{Center} &  \text{Right} \\\\ 
\hline
1 & 0.24 & 1 & 125 \\\\
2 & -1 & 189 & -8 \\\\
3 & -20 & 2000 & 1+10i
\end{array}
$$

# 矩阵
使用`$$ \begin{matrix} ... \end{matrix} $$`，`\\`表示每一行的结尾，`&`用来分割元素
```MathJax
$$
\begin{matrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{matrix}
$$
```
效果如下：
$$
\begin{matrix}
1 & 0 & 0 \\\\
0 & 1 & 0 \\\\
0 & 0 & 1 \\\\
\end{matrix}
$$

如果需要加括号，可以使用上面提到的符号，除此之外还可以通过将`matrix`替换来实现：

* 替换为`\pmatrix`得到：$ \begin{pmatrix}1 & 0 & 0 \\\\0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ \end{pmatrix} $

* 替换为`\bmatrix`得到：$ \begin{bmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ \end{bmatrix} $

* 替换为`\Bmatrix`得到：$ \begin{Bmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ \end{Bmatrix} $

* 替换为`\vmatrix`得到：$ \begin{vmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ \end{vmatrix} $

* 替换为`\Vmatrix`得到：$ \begin{Vmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ \end{Vmatrix} $

如果想省略一些像，可以使用`\cdots`，`\ddots`，`vdots`，来省略行元素，对角元素和列元素：
```MathJax
$$
\begin{pmatrix}
1 & a_1 & a_1^2 & \cdots & a_1^n \\
1 & a_2 & a_2^2 & \cdots & a_2^n \\
\vdots & \vdots & \ddots & \vdots \\
1 & a_n & a_n^2 & \cdots & a_n^n \\
\end{pmatrix}
$$
```
效果如下：
$$
\begin{pmatrix}
1 & a_1 & a_1^2 & \cdots & a_1^n \\\\
1 & a_2 & a_2^2 & \cdots & a_2^n \\\\
\vdots & \vdots & \ddots & \vdots & \vdots \\\\
1 & a_n & a_n^2 & \cdots & a_n^n \\\\
\end{pmatrix}
$$
如果是增光矩阵，可以使用前面介绍的创建表格是方式来实现：
```MathJax
$$
\left[
\begin{array}{cc|c}
1 & 2 & 3 \\
4 & 5 & 6
\end{array}
\right]
$$
```
$$
\left[
\begin{array}{cc|c}
1 & 2 & 3 \\\\
4 & 5 & 6
\end{array}
\right]
$$
文本段内使用矩阵，则需要使用`\big(\begin{smallmatrix} ... \end{smallmatrix}\bigr)`
```MathJax
$$
\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix} \bigr)
$$
```
$$
\bigl(\begin{smallmatrix} a & b \\\\ c & d \end{smallmatrix} \bigr)
$$


# 方程组
使用`\begin{aray} ... \end{array}`与`\left{ ... \rigth.`配合可以表示方程组：
````MathJax
$$
\left \{
\begin{array}
{c}
a_1 x + b_1 y + c_1 z = d_1 + e_1 \\
a_2 x + b_2 y = d_2 \\
a_3 x + b_3 y + c_3 z = d_3
\end{array}
\right.
$$
```
效果如下：

$$
\left \lbrace
\begin{array}
{c}
a_1 x + b_1 y + c_1 z = d_1 + e_1 \\\\
a_2 x + b_2 y = d_2 \\\\
a_3 x + b_3 y + c_3 z = d_3
\end{array}
\right.
$$

此外，还可以使用`\begin{cases} ... \end{cases}`来表示同样的方程组：
```MathJax
$$
\begin{cases}
a_1 x + b_1 y + c_1 z = d_1 + e_1 \\
a_2 x + b_2 y = d_2 \\
a_3 x + b_3 y + c_3 z = d_3
\end{cases}
$$
```
效果如下：
$$
\begin{cases}
a_1 x + b_1 y + c_1 z = d_1 + e_1 \\\\
a_2 x + b_2 y = d_2 \\\\
a_3 x + b_3 y + c_3 z = d_3
\end{cases}
$$
如果需要对齐方程组中的$=$号，可以使用`\begin{aligned} ... \end{aligned}`：
```MathJax
$$
\left\{
\begin{aligned}
a_1 x + b_1 y + c_1 z &= d_1 + e_1 \\
a_2 x + b_2 y &= d_2 \\
a_3 x + b_3 y + c_3 z &= d_3
\end{aligned}
\right.
$$
```
效果如下：
$$
\left\lbrace
\begin{aligned}
a_1 x + b_1 y + c_1 z &= d_1 + e_1 \\\\
a_2 x + b_2 y &= d_2 \\\\
a_3 x + b_3 y + c_3 z &= d_3
\end{aligned}
\right.
$$
如果需要对齐等号和项，可以使用`\begin{array} {列样式} ... \end{array}`：
```MathJax
$$
\left\{
\begin{array}
{l l}
a_1 x + b_1 y + c_1 z &= d_1 + e_1 \\
a_2 x + b_2 y &= d_2 \\
a_3 x + b_3 y + c_3 z &= d_3
\end{array}
\right.
$$
```
效果如下：
$$
\left\lbrace
\begin{array}
{l l}
a_1 x + b_1 y + c_1 z &= d_1 + e_1 \\\\
a_2 x + b_2 y &= d_2 \\\\
a_3 x + b_3 y + c_3 z &= d_3
\end{array}
\right.
$$

# 分类表达式
有些时候，定义函数需要分情况给出表达式，可以使用`$\begin{cases} ... \end{cases}$`。其中，`\\`用来分类，`&`用来表示要对齐的位置。
```MathJax
$$
f(n) =
\begin{cases}
n/2 , & \text{if $n$ is over} \\
3n + 1 & , \text{if $n$ is odd }
\end{cases}
$$
```
$$
f(n) =
\begin{cases}
n/2 , & \text{if $n$ is over} \\\\
3n + 1 & , \text{if $n$ is odd }
\end{cases}
$$
如果想要更多的竖直空间，可以用`\\[2ex]` （3ex，4ex也可以，1ex相当于原始距离）代替 `\\`：
```MathJax
$$
f(n) =
\begin{cases}
n/2 , & \text{if $n$ is over} \\[2ex]
3n + 1 & , \text{if $n$ is odd }
\end{cases}
$$
```
$$
f(n) =
\begin{cases}
\frac {n} {2} , & \text{if $n$ is over} \\\\[2ex]
3n + 1 & , \text{if $n$ is odd }
\end{cases}
$$
上述公式的括号也可以移动到右侧，不过需要使用`$\begin{arry} ... \end{arry}$`来实现：
```MathJax
$$
\left.
\begin{array}
{l}
\text{if $n$ is even:} & n/2 \\[5ex]
\text{if $n$ is odd:}  & 3n+1
\end{array}
\right \rbrace
=f(n)
$$
```
效果如下：
$$
\left.
\begin{array}
{l}
\text{if $n$ is even:} & n/2 \\\\[5ex]
\text{if $n$ is odd:}  & 3n+1
\end{array}
\right \rbrace
=f(n)
$$

# 绝对值和模
* `\lvert`，`\rvert`用来表示绝对值。例如`$\lvert x \rvert$` 表示：$\lvert x \rvert$
* `\lVert`，`\rVert`用来表示绝对值。例如`$\lVert x \rVert$` 表示：$\lVert x \rVert$

# 高亮
为了显著表示某公式，可以使用`\bbox`来高亮表达式：
```MathJax
$$
\bbox[yellow]
{
e^x = \lim_{n \to \infty} \left( 1 + \frac {x} {n} \right) ^n
\qquad (1)
}
$$
```
效果如下：
$$
\bbox[yellow]
{
e^x = \lim_{n \to \infty} \left( 1 + \frac {x} {n} \right) ^n
\qquad (1)
}
$$

```MathJax
$$
\bbox[border:2px solid red]
{
e^x = \lim_{n \to \infty} \left( 1 + \frac {x} {n} \right) ^n
\qquad (1)
}
$$
```
效果如下：
$$
\bbox[border:2px solid red]
{
e^x = \lim_{n \to \infty} \left( 1 + \frac {x} {n} \right) ^n
\qquad (1)
}
$$





# 颜色
| 代码  |  效果 |
| :------------: | :------------: |
|  `$\color{black}{Hello World!}$` |  $\color{black}{Hello World!}$ |
|  `$\color{gray}{Hello World!}$`  |  $\color{gray}{Hello World!}$ |
|  `$\color{silver}{Hello World!}$`  | $\color{silver}{Hello World!}$  |
|  `$\color{white}{Hello World!}$`  |  $\color{white}{Hello World!}$ |
|  `$\color{maroom}{Hello World!}$`  | $\color{maroom}{Hello World!}$ |
|  `$\color{red}{Hello World!}$`  | $\color{red}{Hello World!}$  |
|  `$\color{yellow}{Hello World!}$`  |  $\color{yellow}{Hello World!}$ |
|  `$\color{lime}{Hello World!}$`  | $\color{lime}{Hello World!}$  |
|  `$\color{olive}{Hello World!}$`  | $\color{olive}{Hello World!}$  |
|  `$\color{green}{Hello World!}$`  | $\color{green}{Hello World!}$  |
|  `$\color{teal}{Hello World!}$`  |  $\color{teal}{Hello World!}$ |
|  `$\color{aqua}{Hello World!}$`  |  $\color{aqua}{Hello World!}$ |
|  `$\color{blue}{Hello World!}$`  |  $\color{blue}{Hello World!}$ |
|  `$\color{navy}{Hello World!}$`  |  $\color{navy}{Hello World!}$ |
|  `$\color{purple}{Hello World!}$`  | $\color{purple}{Hello World!}$  |
|  `$\color{fuchsia}{Hello World!}$`  | $\color{fuchsia}{Hello World!}$  |
