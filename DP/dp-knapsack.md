## xcpc 基础补全计划：动态规划篇
### Day 1: 背包DP (1)

### 前言
由于作者刚被抽象代数拷打了整个学期，现在更喜欢借助类似于 **映射** 之类的较为形式化的工具思考问题，所以为了让我能够理解相关知识点，我将会更喜欢用相对形式化的数学语言来写这篇笔记。

笔记中可能会经常提到 **映射** 这个词，不过其实在许多问题里映射的对象都是数集，所以很多时候可以完全把 **映射** 简单理解为数学中很常规的 **函数** ，甚至也可以理解为程序中的 **函数** ，毕竟本质上也是输入某些东西，最后输出一个结果。

官方的动态规划基础教程对这一类问题有着以下描述：
- 最优子结构：子问题的最优解必然是原问题最优解的一部分；
- 无后效性：已经求解过的子问题，不再受到后续决策的影响；
- 子问题重叠：如果有大量重叠的子问题，可以用一个空间将已经求解过的子问题的最优解存储下来，从而不需要重复求解。

动态规划的基本思想一般是先做出 **最简单的** 子问题，然后利用 **规模较小** 的子问题的结果，推出与它相关的 **规模稍大一点** 的问题的结果。如果层层递推上去能够得到原问题（规模很大的问题），也就解决了原问题。

#### 选择性阅读
我们发现这其实与 **数学归纳法** 的想法几乎一致：首先要有一个归纳基础（例如 $n=0$ 时的情形，对应到动态规划里就是最简单的子问题），然后假设 $n = k-1$ 时成立，证明 $n = k$ 时成立（从小规模问题推出大规模问题）。有些时候这种归纳不是单变量的，比如接下来的 **0-1 背包问题** 就需要物品数 $n$ 和体积 $V$ 的二维归纳，但大致的想法也和一维情形是非常类似的。

二维参数 $(m,n)$ 的数学归纳如何实现？其实也很简单，首先固定 $m = 0$，利用单变量的数学归纳法证明这个条件下所有 $n$ 成立，这样就有了 $m = 0$ 这样一个归纳基础。然后假设 $m = k-1$ 时问题对所有 $n$ 成立，再证明 $m = k$ 时所有 $n$ 成立即可。

动态规划问题有许多类型，例如线性DP，背包DP，树形DP...... 不过刚开始写这篇笔记时的我很多东西也还不会，所以不少东西目前仅听过一个名词，但还并不了解。下面我们将从背包DP开始进行动态规划部分的学习。

### 1.1 0-1 背包问题
#### 问题描述
给定 $n$ 个物品与容量为 $V$ 的背包，每件物品有 $w_i$ 的价值与 $v_i$ 的体积。我们需要在放入背包的物品总体积不超过 $V$ 的情况下，使得物品总价值最高。

#### 题解
我们来从映射的角度去理解这个问题。首先我们可以看到，这个问题的最终答案仅取决于以下参数：$n$, $V$, $(w_1, ..., w_n)$, $(v_1, v_2, ..., v_n)$. 只要这些参数完全给定，那么这个问题就会有唯一的答案。用形式化一点的语言描述，这是一个 **从参数到答案** 的映射：

$$\begin{aligned}
F: \mathbb{Z}^{2n+2} & \rightarrow \mathbb{Z} \\
(n,V,w_1,...w_n,v_1,...,v_n) & \mapsto ans 
\end{aligned}$$

说得直白一点，问题的最终答案仅取决于这些参数。保持问题描述中所有的 **汉字** 不动，只是把 **参数** 替换一下，就会得到一个同样完整的问题。

##### 设计状态
面对一个具体的问题，我们怎么样能够让这种映射具有 **可递推** 的属性呢？首先这 $n$ 件物品的价格和体积都是固定的，它们并不会在递推的过程中发生变化。所以我们不妨构造 **更小规模** 的子问题如下：所有物品的价格和体积固定的情况下，只能选择前 $i$ 件物品，而且背包的体积 $j$ 不超过原先的体积，也就是 $i \leq n, j \leq V$. 于是一个子问题就是下面这样的映射（我们把这个映射取名为 $dp$，这样就能和后续状态转移方程里的变量对上名字）：

$$\begin{aligned}
dp: \mathbb{Z}^{2} & \rightarrow \mathbb{Z} \\
(i, j) & \mapsto ans 
\end{aligned}$$

在这里，由 $(i,j)$ 所决定的子问题通常被称为 **状态** ，最终的原问题其实就是 $(n,V)$ 所对应的状态。

##### 状态转移
第一步是考虑最简单的状态，也就是初始化。

在这个问题中，最简单的状态当然是 $i = 0$ 时对应的状态。考虑前 $0$ 件物品，既然什么都没有选，最高价值当然是 $0$，所以子问题 $(0,j)$ 的最优解都是 $0$，即 $dp[0][j] = 0$, $j = 0,1,2,...,V$.

接下来考虑如何进行状态之间的转移。对于任意的状态 $(i,j)$，假如考虑前 $i-1$ 件物品的所有子问题 $(i-1,j')$ 已经求出了最优解 $(j' = 0,1,2,...,V)$，那么状态 $(i,j)$ 就可以由以下两个状态转移过来：

- 不放第 $i$ 件物品，那么 $dp[i-1][j] \rightarrow dp[i][j]$；
- 放第 $i$ 件物品，那么 $dp[i-1][j-v_i] + w_i \rightarrow dp[i][j]$.

我们要使状态 $(i,j)$ 拥有最优的结果，那么自然是从上面两种转移中选取最优的一个，也就得到了最后的状态转移方程

$$\begin{aligned}
dp[i][j] = \max(dp[i-1][j], dp[i-1][j-v_i]+w_i), \\ i = 1,2,...,n, j = 0,1,...,V.
\end{aligned}$$

##### 证明转移符合动态规划的条件
对规模为 $i$ 件物品的子问题，其最优解必然要从规模 $i-1$ 问题的最优解转移而来。因为根据我们的状态转移，如果规模 $i-1$ 时不是最优，只需要取更优的那一个就可以让转移到 $i$ 的结果更大，因此必须保证用子问题的最优解来进行后续转移，这也就是最优子结构；

当我们转移到 $i$ 时，前面所有规模小于 $i$ 的问题的结果只会被调用，而不会被修改，换句话说就是已经求解过的子问题不再受后续状态影响，这就是无后效性；

子问题 $(i-1,j)$ 可能直接转移到 $(i,j)$，也可能将结果加上 $w_i$ 后转移到 $(i, j+v_i)$，两种转移分别对应着不选择/选择第 $i$ 件物品。于是子问题可能被重复两次使用，这时我们用了 $dp$ 数组将求解过的最优解存下来，这就是子问题重叠。

##### 滚动数组优化
在这里我们需要注意，$dp[i-1][j]$ 的结果仅仅会在转移到 $i$ 时被直接用到，而不会在物品数更大的情况被使用，因此我们可以压缩掉关于物品的那一维，直接将状态转移方程改写成以下的形式：

$$ dp[j] = \max(dp[j], dp[j-v_i]+w_i),\  j = V,V-1,...,v_i+1, v_i. $$

这里的等号表示将右端的结果赋值给左端。我们从 $V$ 开始倒序遍历到 $v_i$，因为体积小于 $v_i$ 的背包必然装不下第 $i$ 件物品，只能保持原来考虑前 $i-1$ 件物品时的结果不变。完成这个遍历过程以后，我们就实现了将规模 $i-1$ 的子问题结果完全转移到规模 $i$ 的子问题上，而没有用到二维数组。

##### 为什么需要倒序遍历？
原先的状态转移是 $dp[i][j] = \max(dp[i-1][j], dp[i-1][j-v_i]+w_i)$. 

如果我们是顺序遍历，那么 $dp[i-1][j-v_i]$ 的值就会提前被覆盖为 $dp[i][j-v_i]$，从而会让我们使用错误的状态进行转移。而倒序遍历不会先覆盖掉体积更小的状态，从而能够正确地进行转移。

##### 为什么需要滚动数组优化？
一道题比较常见的限制是 $Time \leq 2s, Memory \leq 512MB.$ 假设不用滚动数组，我们会发现时间复杂度与空间复杂度均为 $O(nV)$。$2s$ 时限大约相当于 $1e9$ 次运算，但 $512MB$ 内存大约只有 $128M \approx 1e8$ 个 $int$，空间会比时间先爆，所以要用滚动数组压缩一维，从而使得空间复杂度可接受。

#### 模板题
- 洛谷 P1048

#### 模板代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int v[100005], w[100005], dp[100005];
signed main(){
	int n,V; cin >> n >> V;
	for(int j = 0; j <= V; j++) dp[j] = 0;
	for(int i = 1; i <= n; i++){
		cin >> v[i] >> w[i];
		for(int j = V; j >= v[i]; j--) dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
	}
	cout << dp[V];
	return 0; 
}
```

### 1.2 完全背包问题
#### 问题描述
给定 $n$ 种物品与容量为 $V$ 的背包，每种物品有 $w_i$ 的价值与 $v_i$ 的体积，且每种物品有无限个。我们需要在放入背包的物品总体积不超过 $V$ 的情况下，使得物品总价值最高。

#### 题解
这个问题与 0-1 背包的唯一区别在于，这里每种物品可以被任意次选择，而 0-1 背包每种物品只能选一次。我们同样设计和 0-1 背包同样的状态，假设仅考虑前 $i$ 件物品，背包体积为 $j$，朴素的做法是考虑枚举第 $i$ 件物品的选择次数 $k$，就有状态转移方程

$$ dp[i][j] = \max_{k = 0}^{+\infty}(dp[i-1][j- k\cdot v_i] + k\cdot w_i) $$

然而实际上我们可以进行一个简单优化。假如说当前状态 $(i,j)$ 的最优解选择了 $k$ 次物品 $i$，必然有 $(i,j-v_i)$ 的最优解选择了 $k-1$ 次物品 $i$，因为在此之前的物品早已被充分考虑，所以只有物品 $i$ 会对状态的转移产生影响，因此 $dp[i][j]$ 要么从 $dp[i-1][j]$ 转移过来，要么从 $dp[i][j-v_i]$ 转移过来，于是有状态转移方程：

$$ dp[i][j] = \max(dp[i-1][j], dp[i][j-v_i]).$$

在使用滚动数组优化时，我们只需要把 0-1 背包中的倒序遍历改为正序遍历即可。这是因为我们的状态转移方程中，第二个转移状态实际上用到的是 **被覆盖后的结果** ，于是最后的状态转移方程改写为

$$ dp[j] = \max(dp[j], dp[j-v_i] + w_i), j = v_i, v_i+1, ..., V.$$

#### 模板题
- 洛谷 P1616

#### 模板代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int v[100005], w[100005], dp[10000007];
signed main(){
	int n,V; cin >> n >> V;
	for(int j = 0; j <= V; j++) dp[j] = 0;
	for(int i = 1; i <= n; i++){
		cin >> v[i] >> w[i];
		for(int j = v[i]; j <= V; j++) dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
	}
	cout << dp[V];
	return 0; 
}
```

### 1.3 多重背包问题
#### 问题描述
给定 $n$ 种物品与容量为 $V$ 的背包，每种物品有 $w_i$ 的价值与 $v_i$ 的体积，且每种物品有 $k_i$ 个。我们需要在放入背包的物品总体积不超过 $V$ 的情况下，使得物品总价值最高。

#### 题解
首先一种朴素的想法是：把第 $i$ 种物品一个一个地拆出来，看成 $k_i$ 件单独的物品，于是变成了一个物品总数为 $\sum k_i$ 的 0-1 背包问题，这样的时间复杂度是 $O(V\sum k_i)$。

##### 二进制优化
考虑到任意范围在 $[0, 2^m-1]$ 的数，均可以由 $2^0, 2^1,...,2^{m-1}$ 这些数选择 $0$ 或 $1$ 次求和表示出来，所以如果第 $i$ 种物品的数量 $k_i \in [2^m-1, 2^{m+1}-1]$, 我们可以先将它拆分为由 $2^p$ 个物品 $i$ 捆绑而成的大物品，$p = 0,1,...,m-1$，再将剩余部分捆绑起来作为一个物品。例如：

- $15 = 1 + 2 + 4 + 8$；
- $38 = 1 + 2 + 4 + 8 + 16 + 7$。

这样我们就能用上面的组合表示选择任意数量的物品 $i$。每个捆绑过的物品只有选或不选两种，所以又变成了一个 0-1 背包问题，但此时等效的物品数量已经降到了 $O(log_{2}(k_i))$。

#### 模板题
- 洛谷 P1776

#### 模板代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int v[100005], w[100005], dp[100005];
signed main(){
	int n,V; cin >> n >> V;
	for(int j = 0; j <= V; j++) dp[j] = 0;
	int cnt = 1;
	for(int i = 1; i <= n; i++){
		int worth, volume, k;
		cin >> worth >> volume >> k;
		int tmp = 1;
		while(k > tmp){
			v[cnt] = volume * tmp;
			w[cnt] = worth * tmp;
			cnt++, k -= tmp, tmp *= 2;
		}
		if(k > 0){
			v[cnt] = volume * k;
			w[cnt] = worth * k;
			cnt++, k = 0;
		}
	}
	for(int i = 1; i < cnt; i++){
		for(int j = V; j >= v[i]; j--) dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
	}
	cout << dp[V];
	return 0; 
}
```

### 1.4 背包问题变种
#### 1.4.1 求恰好装满时的最大价值
##### 模板题 Codeforces 189A: Cut Ribbons
给定长度为 $n$ 的绳子，要将它切成若干段绳子，每段绳子的长度只能是 $a,b,c$ 中的其中一种。我们要求最终能恰好将绳子切完，且段数尽可能多，求最大段数。

这题实际上是一个要求恰好装满的完全背包问题。我们可以把长度 $n$ 看作背包的体积，$a,b,c$ 看作三个物品的体积，而每个物品的价值都是 $1$，这样最后求出的价值就是可分割的最多绳子段数。

实际上我们仍然可以沿用最开始的状态设计，但是这里有一个问题：某些状态由于永远无法被达到，所以是不可转移的。例如：要求将绳子切为 $2,3,4$ 的若干段，则长度为 $1$ 的绳子永远无法满足这个条件，因此我们只能从可到达的状态开始进行转移。

考虑滚动数组压缩过后的状态转移方程：$ dp[j] = \max(dp[j], dp[j-v_i] + w_i)$，只要两种状态中有至少一种有效，就能够完成转移。因此我们不妨考虑将无效状态的值初始化为 $-\infty$（实际操作时只需要初始化为 $-1e18$ 即可）。在 $i = 0$ 的情形，只有 $dp[0][0] = 0$ 是一个有效状态，剩余的 $dp[0][j]$ 均是无效的。这样只要求出的 $dp[j]$ 非负，就说明它完成了一次有效的转移，而如果是负数重新置为 $-\infty$ 即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
const int inf = 1e18;

int v[4], dp[100005];
signed main(){
	int n; cin >> n;
	for(int i = 1; i <= 3; i++) cin >> v[i];
	for(int i = 1; i <= n; i++) dp[i] = -inf;
	dp[0] = 0;
	for(int i = 1; i <= 3; i++){
		for(int j = v[i]; j <= n; j++){
			dp[j] = max(dp[j], dp[j-v[i]] + 1);
			if(dp[j] < 0) dp[j] = -inf;
		}
	}
	if(dp[n] > 0) cout << dp[n];
	else cout << 0 << "\n";
	return 0;
}
```

#### 1.4.2 求恰好装满的方案数
##### 模板题 Codeforces 474D: Flowers
现在有许多红花和白花连成一串，我们要把这串花从头到尾顺着吃掉，每次只允许吃 $1$ 朵红花或连续的 $k$ 朵白花。给定 $k$ 进行 $t$ 组询问，每组询问求所有长度在区间 $[a,b]$ 的花串当中，能够被刚好吃完的花串数量，对 $1e9+7$ 取模。

只需设计 $dp[i]$: 长度为 $i$ 的刚好能够被吃完的花串数量。如果我们已经统计完所有长度小于 $i$ 的合法花串数，那么长为 $i$ 的合法花串恰好可以由以下两种情形得到：

- 在长为 $i-1$ 的合法花串后放一朵红花；
- 在长为 $i-k$ 的合法花串后放连续的 $k$ 朵白花。

长为 $0$ 的串恰有 $1$ 种方案被吃完，即 $dp[0] = 0$，于是得到状态转移方程 $dp[i] = dp[i-1] + dp[i-k]$, 当 $i-k < 0$ 时 $dp[i-k] = 0$.

容易发现该状态转移方程对多个物品时的情形仍有效。

对于多组询问，由于它们给定的 $k$ 相同，只需首先预处理 $dp$ 数组，作 $dp$ 数组的前缀和 $sum$，然后对每组 $(a,b)$ 返回 $sum[b] - sum[a-1]$，注意取模即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int mod = 1e9 + 7;
 
int dp[100005];
int sum[100005];
 
void solve(int k){
	for(int i = 0; i <= 100000; i++) dp[i] = 0, sum[i] = 0;
	dp[0] = 1;
	for(int i = 1; i <= 100000; i++){
		dp[i] = dp[i-1], dp[i] %= mod;
		if(i >= k) dp[i] += dp[i-k], dp[i] %= mod;
		sum[i] = sum[i-1] + dp[i];
		sum[i] %= mod;
	}
}

signed main(){
	ios::sync_with_stdio(false);
	cin.tie(0);
	int t,k; cin >> t >> k;
	solve(k);
	while(t--){
		int a,b; cin >> a >> b;
		int res = sum[b];
		res -= sum[a-1], res += mod, res %= mod;
		cout << res << "\n";
	}
	return 0;
}
```

### Day 2: 背包DP (2)
#### 1.4.3 模意义下的背包问题
##### 练习题 Codeforces 1105C: Ayoub and Lost Array
给定整数 $n$ 和区间 $[l,r]$，要在区间内随意选择 $n$ 个数（可重复），且它们的和恰好模 $3$ 余 $0$，求选择方案数对 $1e9+7$ 取模。

我们只需统计 $[l,r]$ 中分别有多少数模 $3$ 余 $0,1,2$ 即可，用数组 $cnt[3]$ 进行统计。

设 $dp[i][j]$: 考虑前 $i$ 个数选择后，已选择的数求和模 $3$ 余 $j$。初始状态 $dp[0][0] = 1, dp[0][1] = 0, dp[0][2] = 0$，那么有状态转移：

$$ dp[i][0] = dp[i-1][0] * cnt[0] + dp[i-1][1] * cnt[2] + dp[i-1][2] * cnt[1],$$

$$ dp[i][1] = dp[i-1][0] * cnt[1] + dp[i-1][1] * cnt[0] + dp[i-1][2] * cnt[2],$$

$$ dp[i][2] = dp[i-1][0] * cnt[2] + dp[i-1][1] * cnt[1] + dp[i-1][2] * cnt[0].$$

注意转移时取模，最后输出 $dp[n][0]$ 即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int mod = 1e9 + 7;
int cnt[3], dp[200005][3];

signed main(){
	int n,l,r; cin >> n >> l >> r;
	cnt[0] = cnt[1] = cnt[2] = 0;
	while(l % 3 != 0) cnt[l % 3]++, l++;
	while(r % 3 != 2 && r % 3 != -1) cnt[r % 3]++, r--;
	for(int i = 0; i < 3; i++) cnt[i] += ((r-l+1)/3);
	
	dp[0][0] = 1, dp[0][1] = 0, dp[0][2] = 0;
	for(int i = 1; i <= n; i++){
		dp[i][0] = dp[i-1][0] * cnt[0] + dp[i-1][1] * cnt[2] + dp[i-1][2] * cnt[1];
		dp[i][1] = dp[i-1][0] * cnt[1] + dp[i-1][1] * cnt[0] + dp[i-1][2] * cnt[2];
		dp[i][2] = dp[i-1][0] * cnt[2] + dp[i-1][1] * cnt[1] + dp[i-1][2] * cnt[0];
		dp[i][0] %= mod, dp[i][1] %= mod, dp[i][2] %= mod;
	}
	
	cout << dp[n][0];
	return 0;
}
```

#### 1.4.4 混合背包问题
##### 练习题 洛谷 P1833: 樱花
给定一个起始时间和一个终止时间，要看庭院里的 $n$ 棵樱花树。每看一棵樱花树要花 $t_i$ 分钟，获得美学值 $c_i$，最多允许看 $p_i$ 次，其中 $p_i = 0$ 表示可以看无数次，$p_i$ 为有限值表示可以看的最多次数。求从起始时间开始，到终止时间前看樱花可以获得的最大美学值。

这题就是前面所讲的三种背包的混合，只需先把可看多次的樱花树用二进制优化的方式拆成许多可以看一次的樱花树，就变成了完全背包和 0-1 背包的混合。在状态转移时，如果第 $i$ 棵树是 0-1 背包，就倒序遍历；如果是完全背包就顺序遍历。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int t[100005], c[100005], p[100005];
int dp[2003];

signed main(){
	int h1,m1,h2,m2,n;
	char ch;
	cin >> h1 >> ch >> m1 >> h2 >> ch >> m2 >> n;
	int minute = 60*h2 - 60*h1 + m2 - m1;
	int cnt = 1;
	for(int i = 1; i <= n; i++){
		int t1, c1, p1;
		cin >> t1 >> c1 >> p1;
		if(p1 == 1 or p1 == 0){
			t[cnt] = t1, c[cnt] = c1, p[cnt] = p1, cnt++;
		}else{
			int tmp = 1;
			while(p1 > tmp){
				t[cnt] = t1 * tmp; c[cnt] = c1 * tmp;
				p[cnt] = 1, p1 -= tmp, tmp *= 2, cnt++;
			}
			if(p1 > 0){
				t[cnt] = t1 * p1; c[cnt] = c1 * p1;
				p[cnt] = 1, p1 = 0, cnt++;
			}
		}
	}
	
	for(int i = 0; i <= 2000; i++) dp[i] = 0;
	for(int i = 1; i < cnt; i++){
		if(p[i] == 1){
			for(int j = minute; j >= t[i]; j--){
				dp[j] = max(dp[j], dp[j-t[i]] + c[i]);
			}
		}else{
			for(int j = t[i]; j <= minute; j++){
				dp[j] = max(dp[j], dp[j-t[i]] + c[i]);
			}
		}
	}
	cout << dp[minute];
	return 0;
}
```

##### 练习题 Codeforces 106C: Buns
小 $L$ 有 $n$ 克面粉和 $m$ 种馅料，他可以制作以下 $m$ 种馅饼：

- 馅料 $i$ 有 $a_i$ 克，每做一个馅饼 $i$ 需要 $b_i$ 克对应馅料和 $c_i$ 克面粉，每个能卖 $d_i$ 元；
- 只用 $c_0$ 克面粉做一个没有馅的饼，每个能卖 $d_0$ 元。

根据上面的限制条件，求小 $L$ 最多能卖多少钱。

这也是混合背包问题。面粉的数量相当于背包的体积，卖的钱相当于价值，每个馅饼 $i$ 根据馅料的限制转化成了物品的重数，不放馅料的饼相当于一个可以选无数次的物品。然后根据上一题的思路对不同类型的物品分别写转移方程即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int dp[1003], v[103], w[103];

signed main(){
	int n,m,c0,d0; cin >> n >> m >> c0 >> d0;
	int cnt = 1;
	for(int i = 1; i <= m; i++){
		int a,b,c,d; cin >> a >> b >> c >> d;
		int q = a/b, tmp = 1;
		while(q > tmp){
			v[cnt] = c * tmp, w[cnt] = d * tmp;
			q -= tmp, tmp *= 2, cnt++;
		}
		if(q > 0){
			v[cnt] = c * q, w[cnt] = d * q;
			q = 0, cnt++;
		}
	}
	
	for(int i = 0; i <= n; i++) dp[i] = 0;
	for(int i = 1; i < cnt; i++){
		for(int j = n; j >= v[i]; j--) dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
	}
	for(int j = c0; j <= n; j++) dp[j] = max(dp[j], dp[j-c0] + d0);
	cout << dp[n];
	return 0;
} 
```

#### 1.4.5 分组背包
##### 练习题 洛谷 P1757: 通天之分组背包
现有一个可承载体积为 $m$ 的背包，一共有 $n$ 件物品，第 $i$ 件物品的体积是 $a_i$，价值是 $b_i$，属于第 $c_i$ 组物品。要求每组物品至多选择一个，求背包能够装下物品的最大价值。

我们仍然考虑 $dp[i][j]$: 只考虑前 $i$ 组物品，背包容量为 $j$ 时的最大价值。我们给第 $i$ 组中每个物品分别编号 $(i,k), k = 1,2,...,cnt[i]$，其中 $cnt[i]$ 表示第 $i$ 组的物品数量。为了保证每组只选一个，我们将从第 $i-1$ 组到第 $i$ 组的转移视为第 $1$ 层循环，第 $2$ 层循环中首先要遍历背包容积 $j = m, m-1,..., 0$，然后在第 $3$ 层循环中遍历物品 $(i,1),...,(i, cnt[i])。同样利用滚动数组压缩，我们有状态转移方程：

$$ dp[j] = max(dp[j], dp[j - v[i][k]] + w[i][k]) $$

其中先让 $k$ 遍历完所有物品下标，再在外层倒序遍历所有背包容积。$v[i][k], w[i][k]$ 分别表示第 $i$ 组第 $k$ 个物品的体积与价值，最终我们就完成了分组背包。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int v[101][1001], w[101][1001], cnt[101];
int dp[1001];

signed main(){
	int m,n; cin >> m >> n;
	for(int i = 0; i <= 100; i++) cnt[i] = 0;
	for(int i = 0; i <= m; i++) dp[i] = 0;
	for(int i = 1; i <= n; i++){
		int a,b,c; cin >> a >> b >> c;
		cnt[c]++;
		v[c][cnt[c]] = a, w[c][cnt[c]] = b;
	}
	
	for(int i = 1; i <= 100; i++){
		if(cnt[i] == 0) continue;
		for(int j = m; j >= 0; j--){
			for(int k = 1; k <= cnt[i]; k++){
				if(j >= v[i][k]){
					dp[j] = max(dp[j], dp[j - v[i][k]] + w[i][k]);
				}
			}
		}
	}
	
	cout << dp[m];
}
```

##### 练习题 Codeforces 148E: Porcelain
有 $n$ 个货架，每个货架上有 $c_i$ 件物品，每件物品的价值依次是 $w_{i,1}, ..., w_{i,c_i}$。每次拿物品时必须从货架的最左端或最右端拿，一共要拿 $m$ 件，保证所有货架上的物品数量大于等于 $m$，求能够拿到的物品的最大价值。

可以发现，我们要在某个货架上拿到 $k$ 个物品，它的价值必然由该货架上的某段前缀和与某段后缀和相加得到。所以我们可以枚举在该货架上拿的物品数 $k = 1,2,...,c_i$，并枚举拿前面的 $k_1 = 0,1,...,k$ 个物品，从中选择拿 $k$ 个物品最优的情况存下来，这样对于单个货架，拿 $0,1,...c_i$ 个物品就变成了分组背包当中的一组物品，接下来使用分组背包模板即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int a[102][102];
int presum[102][102], sufsum[102][102];
int w[102][102], q[102];
int dp[10004];

void init(int n, int m){
	for(int i = 1; i <= n; i++){
		for(int j = 0; j <= 100; j++) w[i][j] = 0;
		q[i] = 0;
	}
	for(int i = 0; i <= m; i++) dp[i] = 0;
}

signed main(){
	int n,m; cin >> n >> m;
	init(n, m);
	for(int i = 1; i <= n; i++){
		int cnt; cin >> cnt;
		q[i] = cnt, presum[i][0] = 0;
		for(int j = 1; j <= cnt; j++){
			cin >> a[i][j];
			presum[i][j] = presum[i][j-1] + a[i][j];
		}
		sufsum[i][cnt+1] = 0;
		for(int j = cnt; j >= 1; j--) sufsum[i][j] = sufsum[i][j+1] + a[i][j];
	}
	
	for(int i = 1; i <= n; i++){
		for(int j = 1; j < q[i]; j++){
			for(int k = 0; k <= j; k++){
				int q1 = k, q2 = q[i] + 1 - (j - k);
				w[i][j] = max(w[i][j], presum[i][q1] + sufsum[i][q2]);
			}
		}
		w[i][q[i]] = presum[i][q[i]]; 
	}
	
	for(int i = 1; i <= n; i++){
		for(int j = m; j >= 0; j--){
			for(int k = 1; k <= q[i]; k++){
				if(j >= k) dp[j] = max(dp[j], dp[j-k] + w[i][k]);
			}
		}
	}
	cout << dp[m];
}
```


#### 1.4.6 依赖背包
##### 练习题 洛谷 P1064: 金明的预算方案
金明共有 $n$ 元预算，等待购买的物品有 $m$ 件。其中物品分为主件和附件两种，如果想要购买附件，就必须先购买它对应的主件；购买主件没有必须购买附件的要求。每个主件可以有 $0,1,2$ 个附件。

第 $i$ 件商品的价格为 $v_i$，重要程度为 $p_i$，并且是 $q_i$ 的附件（如果 $q_i = 0$ 表示它本身就是主件）。一件商品的价值 $w_i$ 定义为价格乘以重要程度 $v_i \cdot p_i$，其中 $1 \leq p_i \leq 5$。求在上面的限制条件下，金明拥有的预算可以买到商品的最大价值。

考虑主件 $a$ 和它的附件 $a_1, a_2$，我们不难发现对于这一套商品，有以下几种购买方式：

- $a$ $(00)$;
- $a, a1$ $(01)$;
- $a, a2$ $(10)$;
- $a, a1, a2$ $(11)$;

以上这几种购买方式是互斥的，至多只能选择其中一种。如果我们把每种购买方式分别捆绑成一个大商品，并且让它们放在一组内，这就是前面提到的分组背包模型。考虑如何在组内添加商品时，我们可以使用二进制遍历的方式进行，就像上面方案标注的一样。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

struct Node{
	int volume, worth;
}Node[61];

vector<int> group[61];
int cnt[61], w[61][10], v[61][10], dp[32004];
int pow2[61];

void integrate(int i, int j){
	int w1 = Node[group[i][0]].worth, v1 = Node[group[i][0]].volume, j1 = j;
	int digit = 1;
	while(j1){
		if(j1 & 1) w1 += Node[group[i][digit]].worth, v1 += Node[group[i][digit]].volume;
		j1 >>= 1, digit++;
	}
	w[i][j] = w1, v[i][j] = v1;
}

signed main(){
	pow2[0] = 1;
	for(int i = 1; i <= 30; i++) pow2[i] = 2 * pow2[i-1];
	int n,m; cin >> n >> m;
	for(int i = 0; i <= n; i++) dp[i] = 0;
	for(int i = 1; i <= m; i++){
		int v,p,q; cin >> v >> p >> q;
		Node[i].volume = v;
		Node[i].worth = v * p;
		if(q == 0) group[i].insert(group[i].begin(), i);
		else group[q].push_back(i);
	}
	
	for(int i = 1; i <= m; i++){
		if(group[i].size() == 0) continue;
		cnt[i] = pow2[group[i].size() - 1];
		for(int j = 0; j < cnt[i]; j++){
			integrate(i, j);
		}
	}
	
	for(int i = 1; i <= m; i++){
		if(cnt[i] == 0) continue;
		for(int j = n; j >= 0; j--){
			for(int k = 0; k < cnt[i]; k++){
				if(j >= v[i][k]) dp[j] = max(dp[j], dp[j - v[i][k]] + w[i][k]);
			}
		}
	}
	
	cout << dp[n];
}
```

#### 1.4.7 二维费用背包
##### 练习题 Codeforces 543A: Writing Code
现有 $n$ 位程序员一共要写出恰好 $m$ 行代码，第 $i$ 个程序员每一行都恰好有 $a_i$ 个 bug。如果说这些程序员写出的代码的 bug 数量加起来不超过 $b$，则称这是一套好的方案。给定模数 $mod$，求好的方案数对 $mod$ 取模的结果。

我们可以发现这是一个完全背包问题，因为每位程序员写代码的行数并没有限制。设 $dp[i][j][k]$：只考虑前 $i$ 位程序员时，写出的代码恰好 $j$ 行且 bug 数不超过 $k$ 的方案数，则有初始状态 $dp[0][0][k] = 1, k \geq 0$。状态转移方程如下：

$$ dp[i][j][k] = dp[i-1][j][k] + dp[i-1][j-1][k-a[i]]. $$

由于这是完全背包，所以从小状态到大状态进行转移。利用滚动数组我们得到

$$ dp[j][k] = dp[j][k] + dp[j-1][k-a[i]]. $$

注意在转移时取模即可。

##### 通过代码
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

int dp[502][502];
int a[502];

signed main(){
	int n,m,b,mod; cin >> n >> m >> b >> mod;
	for(int i = 0; i <= b; i++) dp[0][i] = 1;
	for(int i = 1; i <= m; i++){
		for(int j = 0; j <= b; j++) dp[i][j] = 0;
	}
	
	for(int i = 1; i <= n; i++) cin >> a[i];
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= m; j++){
			for(int k = a[i]; k <= b; k++){
				dp[j][k] = dp[j][k] + dp[j-1][k-a[i]];
				dp[j][k] %= mod;
			}
		}
	}
	cout << dp[m][b];
}
```
