??.
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??%
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:*
dtype0
?
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_34/gamma
?
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_34/beta
?
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_34/moving_mean
?
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_34/moving_variance
?
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1592/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv2d_1592/kernel
?
&conv2d_1592/kernel/Read/ReadVariableOpReadVariableOpconv2d_1592/kernel*&
_output_shapes
: *
dtype0
x
conv2d_1592/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1592/bias
q
$conv2d_1592/bias/Read/ReadVariableOpReadVariableOpconv2d_1592/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1592/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1592/gamma
?
2batch_normalization_1592/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1592/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1592/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1592/beta
?
1batch_normalization_1592/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1592/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1592/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1592/moving_mean
?
8batch_normalization_1592/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1592/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1592/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1592/moving_variance
?
<batch_normalization_1592/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1592/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1598/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1598/kernel
?
&conv2d_1598/kernel/Read/ReadVariableOpReadVariableOpconv2d_1598/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1598/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1598/bias
q
$conv2d_1598/bias/Read/ReadVariableOpReadVariableOpconv2d_1598/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1598/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1598/gamma
?
2batch_normalization_1598/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1598/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1598/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1598/beta
?
1batch_normalization_1598/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1598/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1598/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1598/moving_mean
?
8batch_normalization_1598/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1598/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1598/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1598/moving_variance
?
<batch_normalization_1598/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1598/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1608/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1608/kernel
?
&conv2d_1608/kernel/Read/ReadVariableOpReadVariableOpconv2d_1608/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1608/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1608/bias
q
$conv2d_1608/bias/Read/ReadVariableOpReadVariableOpconv2d_1608/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1608/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1608/gamma
?
2batch_normalization_1608/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1608/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1608/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1608/beta
?
1batch_normalization_1608/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1608/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1608/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1608/moving_mean
?
8batch_normalization_1608/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1608/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1608/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1608/moving_variance
?
<batch_normalization_1608/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1608/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1612/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1612/kernel
?
&conv2d_1612/kernel/Read/ReadVariableOpReadVariableOpconv2d_1612/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1612/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1612/bias
q
$conv2d_1612/bias/Read/ReadVariableOpReadVariableOpconv2d_1612/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1612/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1612/gamma
?
2batch_normalization_1612/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1612/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1612/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1612/beta
?
1batch_normalization_1612/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1612/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1612/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1612/moving_mean
?
8batch_normalization_1612/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1612/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1612/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1612/moving_variance
?
<batch_normalization_1612/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1612/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1625/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1625/kernel
?
&conv2d_1625/kernel/Read/ReadVariableOpReadVariableOpconv2d_1625/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1625/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1625/bias
q
$conv2d_1625/bias/Read/ReadVariableOpReadVariableOpconv2d_1625/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1625/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1625/gamma
?
2batch_normalization_1625/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1625/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1625/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1625/beta
?
1batch_normalization_1625/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1625/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1625/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1625/moving_mean
?
8batch_normalization_1625/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1625/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1625/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1625/moving_variance
?
<batch_normalization_1625/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1625/moving_variance*
_output_shapes
: *
dtype0

dense_878/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*!
shared_namedense_878/kernel
x
$dense_878/kernel/Read/ReadVariableOpReadVariableOpdense_878/kernel*!
_output_shapes
:???*
dtype0
u
dense_878/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_878/bias
n
"dense_878/bias/Read/ReadVariableOpReadVariableOpdense_878/bias*
_output_shapes	
:?*
dtype0
}
dense_879/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_namedense_879/kernel
v
$dense_879/kernel/Read/ReadVariableOpReadVariableOpdense_879/kernel*
_output_shapes
:	?
*
dtype0
t
dense_879/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_879/bias
m
"dense_879/bias/Read/ReadVariableOpReadVariableOpdense_879/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
SGD/conv2d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameSGD/conv2d_4/kernel/momentum
?
0SGD/conv2d_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_4/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv2d_4/bias/momentum
?
.SGD/conv2d_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_4/bias/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_4/gamma/momentum
?
<SGD/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_4/gamma/momentum*
_output_shapes
:*
dtype0
?
'SGD/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'SGD/batch_normalization_4/beta/momentum
?
;SGD/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_4/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_15/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_15/kernel/momentum
?
1SGD/conv2d_15/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_15/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_15/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameSGD/conv2d_15/bias/momentum
?
/SGD/conv2d_15/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_15/bias/momentum*
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_15/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_15/gamma/momentum
?
=SGD/batch_normalization_15/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_15/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_15/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_15/beta/momentum
?
<SGD/batch_normalization_15/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_15/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_34/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_34/kernel/momentum
?
1SGD/conv2d_34/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_34/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_34/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameSGD/conv2d_34/bias/momentum
?
/SGD/conv2d_34/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_34/bias/momentum*
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_34/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_34/gamma/momentum
?
=SGD/batch_normalization_34/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_34/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_34/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_34/beta/momentum
?
<SGD/batch_normalization_34/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_34/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_1592/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!SGD/conv2d_1592/kernel/momentum
?
3SGD/conv2d_1592/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1592/kernel/momentum*&
_output_shapes
: *
dtype0
?
SGD/conv2d_1592/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1592/bias/momentum
?
1SGD/conv2d_1592/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1592/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1592/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1592/gamma/momentum
?
?SGD/batch_normalization_1592/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1592/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1592/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1592/beta/momentum
?
>SGD/batch_normalization_1592/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1592/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1598/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!SGD/conv2d_1598/kernel/momentum
?
3SGD/conv2d_1598/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1598/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_1598/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1598/bias/momentum
?
1SGD/conv2d_1598/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1598/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1598/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1598/gamma/momentum
?
?SGD/batch_normalization_1598/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1598/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1598/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1598/beta/momentum
?
>SGD/batch_normalization_1598/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1598/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1608/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!SGD/conv2d_1608/kernel/momentum
?
3SGD/conv2d_1608/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1608/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_1608/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1608/bias/momentum
?
1SGD/conv2d_1608/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1608/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1608/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1608/gamma/momentum
?
?SGD/batch_normalization_1608/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1608/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1608/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1608/beta/momentum
?
>SGD/batch_normalization_1608/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1608/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1612/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!SGD/conv2d_1612/kernel/momentum
?
3SGD/conv2d_1612/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1612/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_1612/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1612/bias/momentum
?
1SGD/conv2d_1612/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1612/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1612/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1612/gamma/momentum
?
?SGD/batch_normalization_1612/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1612/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1612/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1612/beta/momentum
?
>SGD/batch_normalization_1612/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1612/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1625/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!SGD/conv2d_1625/kernel/momentum
?
3SGD/conv2d_1625/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1625/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_1625/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1625/bias/momentum
?
1SGD/conv2d_1625/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1625/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1625/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1625/gamma/momentum
?
?SGD/batch_normalization_1625/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1625/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1625/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1625/beta/momentum
?
>SGD/batch_normalization_1625/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1625/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_878/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*.
shared_nameSGD/dense_878/kernel/momentum
?
1SGD/dense_878/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_878/kernel/momentum*!
_output_shapes
:???*
dtype0
?
SGD/dense_878/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameSGD/dense_878/bias/momentum
?
/SGD/dense_878/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_878/bias/momentum*
_output_shapes	
:?*
dtype0
?
SGD/dense_879/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*.
shared_nameSGD/dense_879/kernel/momentum
?
1SGD/dense_879/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_879/kernel/momentum*
_output_shapes
:	?
*
dtype0
?
SGD/dense_879/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameSGD/dense_879/bias/momentum
?
/SGD/dense_879/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_879/bias/momentum*
_output_shapes
:
*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-14
 layer-31
!layer_with_weights-15
!layer-32
"layer-33
#layer-34
$layer_with_weights-16
$layer-35
%layer_with_weights-17
%layer-36
&	optimizer
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+
signatures
 
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
R
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
R
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
R
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
?
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
R
|regularization_losses
}trainable_variables
~	variables
	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter

?decay
?learning_rate
?momentum,momentum?-momentum?3momentum?4momentum?Cmomentum?Dmomentum?Jmomentum?Kmomentum?Vmomentum?Wmomentum?]momentum?^momentum?mmomentum?nmomentum?tmomentum?umomentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum?
 
?
,0
-1
32
43
C4
D5
J6
K7
V8
W9
]10
^11
m12
n13
t14
u15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?
,0
-1
32
43
54
65
C6
D7
J8
K9
L10
M11
V12
W13
]14
^15
_16
`17
m18
n19
t20
u21
v22
w23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
'regularization_losses
(trainable_variables
)	variables
 ?layer_regularization_losses
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
.regularization_losses
/trainable_variables
0	variables
 ?layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
52
63
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
7regularization_losses
8trainable_variables
9	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
;regularization_losses
<trainable_variables
=	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
@trainable_variables
A	variables
 ?layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
L2
M3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
 ?layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Xregularization_losses
Ytrainable_variables
Z	variables
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
_2
`3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
aregularization_losses
btrainable_variables
c	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
eregularization_losses
ftrainable_variables
g	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
 ?layer_regularization_losses
^\
VARIABLE_VALUEconv2d_1592/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_1592/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
oregularization_losses
ptrainable_variables
q	variables
 ?layer_regularization_losses
 
ig
VARIABLE_VALUEbatch_normalization_1592/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_1592/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_1592/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_1592/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
v2
w3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
xregularization_losses
ytrainable_variables
z	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
|regularization_losses
}trainable_variables
~	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
^\
VARIABLE_VALUEconv2d_1598/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_1598/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
ig
VARIABLE_VALUEbatch_normalization_1598/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_1598/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_1598/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_1598/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
_]
VARIABLE_VALUEconv2d_1608/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_1608/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
jh
VARIABLE_VALUEbatch_normalization_1608/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_1608/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_1608/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_1608/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
_]
VARIABLE_VALUEconv2d_1612/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_1612/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
jh
VARIABLE_VALUEbatch_normalization_1612/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_1612/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_1612/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_1612/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
_]
VARIABLE_VALUEconv2d_1625/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_1625/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
jh
VARIABLE_VALUEbatch_normalization_1625/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_1625/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_1625/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_1625/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
][
VARIABLE_VALUEdense_878/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_878/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
][
VARIABLE_VALUEdense_879/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_879/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36

?0
?1
~
50
61
L2
M3
_4
`5
v6
w7
?8
?9
?10
?11
?12
?13
?14
?15
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

v0
w1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUESGD/conv2d_4/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_4/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_4/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'SGD/batch_normalization_4/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_15/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_15/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_15/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_15/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_34/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_34/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_34/gamma/momentumXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_34/beta/momentumWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1592/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1592/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1592/gamma/momentumXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1592/beta/momentumWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1598/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1598/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1598/gamma/momentumXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1598/beta/momentumWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1608/kernel/momentumZlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1608/bias/momentumXlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1608/gamma/momentumYlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1608/beta/momentumXlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1612/kernel/momentumZlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1612/bias/momentumXlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1612/gamma/momentumYlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1612/beta/momentumXlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1625/kernel/momentumZlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1625/bias/momentumXlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1625/gamma/momentumYlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1625/beta/momentumXlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_878/kernel/momentumZlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_878/bias/momentumXlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_879/kernel/momentumZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_879/bias/momentumXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_34/kernelconv2d_34/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_1592/kernelconv2d_1592/biasbatch_normalization_1592/gammabatch_normalization_1592/beta$batch_normalization_1592/moving_mean(batch_normalization_1592/moving_varianceconv2d_1598/kernelconv2d_1598/biasbatch_normalization_1598/gammabatch_normalization_1598/beta$batch_normalization_1598/moving_mean(batch_normalization_1598/moving_varianceconv2d_1608/kernelconv2d_1608/biasbatch_normalization_1608/gammabatch_normalization_1608/beta$batch_normalization_1608/moving_mean(batch_normalization_1608/moving_varianceconv2d_1612/kernelconv2d_1612/biasbatch_normalization_1612/gammabatch_normalization_1612/beta$batch_normalization_1612/moving_mean(batch_normalization_1612/moving_varianceconv2d_1625/kernelconv2d_1625/biasbatch_normalization_1625/gammabatch_normalization_1625/beta$batch_normalization_1625/moving_mean(batch_normalization_1625/moving_variancedense_878/kerneldense_878/biasdense_879/kerneldense_879/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_5508737
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp&conv2d_1592/kernel/Read/ReadVariableOp$conv2d_1592/bias/Read/ReadVariableOp2batch_normalization_1592/gamma/Read/ReadVariableOp1batch_normalization_1592/beta/Read/ReadVariableOp8batch_normalization_1592/moving_mean/Read/ReadVariableOp<batch_normalization_1592/moving_variance/Read/ReadVariableOp&conv2d_1598/kernel/Read/ReadVariableOp$conv2d_1598/bias/Read/ReadVariableOp2batch_normalization_1598/gamma/Read/ReadVariableOp1batch_normalization_1598/beta/Read/ReadVariableOp8batch_normalization_1598/moving_mean/Read/ReadVariableOp<batch_normalization_1598/moving_variance/Read/ReadVariableOp&conv2d_1608/kernel/Read/ReadVariableOp$conv2d_1608/bias/Read/ReadVariableOp2batch_normalization_1608/gamma/Read/ReadVariableOp1batch_normalization_1608/beta/Read/ReadVariableOp8batch_normalization_1608/moving_mean/Read/ReadVariableOp<batch_normalization_1608/moving_variance/Read/ReadVariableOp&conv2d_1612/kernel/Read/ReadVariableOp$conv2d_1612/bias/Read/ReadVariableOp2batch_normalization_1612/gamma/Read/ReadVariableOp1batch_normalization_1612/beta/Read/ReadVariableOp8batch_normalization_1612/moving_mean/Read/ReadVariableOp<batch_normalization_1612/moving_variance/Read/ReadVariableOp&conv2d_1625/kernel/Read/ReadVariableOp$conv2d_1625/bias/Read/ReadVariableOp2batch_normalization_1625/gamma/Read/ReadVariableOp1batch_normalization_1625/beta/Read/ReadVariableOp8batch_normalization_1625/moving_mean/Read/ReadVariableOp<batch_normalization_1625/moving_variance/Read/ReadVariableOp$dense_878/kernel/Read/ReadVariableOp"dense_878/bias/Read/ReadVariableOp$dense_879/kernel/Read/ReadVariableOp"dense_879/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0SGD/conv2d_4/kernel/momentum/Read/ReadVariableOp.SGD/conv2d_4/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_4/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_4/beta/momentum/Read/ReadVariableOp1SGD/conv2d_15/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_15/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_15/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_15/beta/momentum/Read/ReadVariableOp1SGD/conv2d_34/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_34/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_34/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_34/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1592/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1592/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1592/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1592/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1598/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1598/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1598/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1598/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1608/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1608/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1608/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1608/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1612/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1612/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1612/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1612/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1625/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1625/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1625/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1625/beta/momentum/Read/ReadVariableOp1SGD/dense_878/kernel/momentum/Read/ReadVariableOp/SGD/dense_878/bias/momentum/Read/ReadVariableOp1SGD/dense_879/kernel/momentum/Read/ReadVariableOp/SGD/dense_879/bias/momentum/Read/ReadVariableOpConst*m
Tinf
d2b	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_5511019
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_34/kernelconv2d_34/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_1592/kernelconv2d_1592/biasbatch_normalization_1592/gammabatch_normalization_1592/beta$batch_normalization_1592/moving_mean(batch_normalization_1592/moving_varianceconv2d_1598/kernelconv2d_1598/biasbatch_normalization_1598/gammabatch_normalization_1598/beta$batch_normalization_1598/moving_mean(batch_normalization_1598/moving_varianceconv2d_1608/kernelconv2d_1608/biasbatch_normalization_1608/gammabatch_normalization_1608/beta$batch_normalization_1608/moving_mean(batch_normalization_1608/moving_varianceconv2d_1612/kernelconv2d_1612/biasbatch_normalization_1612/gammabatch_normalization_1612/beta$batch_normalization_1612/moving_mean(batch_normalization_1612/moving_varianceconv2d_1625/kernelconv2d_1625/biasbatch_normalization_1625/gammabatch_normalization_1625/beta$batch_normalization_1625/moving_mean(batch_normalization_1625/moving_variancedense_878/kerneldense_878/biasdense_879/kerneldense_879/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1SGD/conv2d_4/kernel/momentumSGD/conv2d_4/bias/momentum(SGD/batch_normalization_4/gamma/momentum'SGD/batch_normalization_4/beta/momentumSGD/conv2d_15/kernel/momentumSGD/conv2d_15/bias/momentum)SGD/batch_normalization_15/gamma/momentum(SGD/batch_normalization_15/beta/momentumSGD/conv2d_34/kernel/momentumSGD/conv2d_34/bias/momentum)SGD/batch_normalization_34/gamma/momentum(SGD/batch_normalization_34/beta/momentumSGD/conv2d_1592/kernel/momentumSGD/conv2d_1592/bias/momentum+SGD/batch_normalization_1592/gamma/momentum*SGD/batch_normalization_1592/beta/momentumSGD/conv2d_1598/kernel/momentumSGD/conv2d_1598/bias/momentum+SGD/batch_normalization_1598/gamma/momentum*SGD/batch_normalization_1598/beta/momentumSGD/conv2d_1608/kernel/momentumSGD/conv2d_1608/bias/momentum+SGD/batch_normalization_1608/gamma/momentum*SGD/batch_normalization_1608/beta/momentumSGD/conv2d_1612/kernel/momentumSGD/conv2d_1612/bias/momentum+SGD/batch_normalization_1612/gamma/momentum*SGD/batch_normalization_1612/beta/momentumSGD/conv2d_1625/kernel/momentumSGD/conv2d_1625/bias/momentum+SGD/batch_normalization_1625/gamma/momentum*SGD/batch_normalization_1625/beta/momentumSGD/dense_878/kernel/momentumSGD/dense_878/bias/momentumSGD/dense_879/kernel/momentumSGD/dense_879/bias/momentum*l
Tine
c2a*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_5511317??!
?
?
:__inference_batch_normalization_1625_layer_call_fn_5510634

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_55069072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?1
"__inference__wrapped_model_5506045
input_15
1model_438_conv2d_4_conv2d_readvariableop_resource6
2model_438_conv2d_4_biasadd_readvariableop_resource;
7model_438_batch_normalization_4_readvariableop_resource=
9model_438_batch_normalization_4_readvariableop_1_resourceL
Hmodel_438_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceN
Jmodel_438_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource6
2model_438_conv2d_15_conv2d_readvariableop_resource7
3model_438_conv2d_15_biasadd_readvariableop_resource<
8model_438_batch_normalization_15_readvariableop_resource>
:model_438_batch_normalization_15_readvariableop_1_resourceM
Imodel_438_batch_normalization_15_fusedbatchnormv3_readvariableop_resourceO
Kmodel_438_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource6
2model_438_conv2d_34_conv2d_readvariableop_resource7
3model_438_conv2d_34_biasadd_readvariableop_resource<
8model_438_batch_normalization_34_readvariableop_resource>
:model_438_batch_normalization_34_readvariableop_1_resourceM
Imodel_438_batch_normalization_34_fusedbatchnormv3_readvariableop_resourceO
Kmodel_438_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8
4model_438_conv2d_1592_conv2d_readvariableop_resource9
5model_438_conv2d_1592_biasadd_readvariableop_resource>
:model_438_batch_normalization_1592_readvariableop_resource@
<model_438_batch_normalization_1592_readvariableop_1_resourceO
Kmodel_438_batch_normalization_1592_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_438_batch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource8
4model_438_conv2d_1598_conv2d_readvariableop_resource9
5model_438_conv2d_1598_biasadd_readvariableop_resource>
:model_438_batch_normalization_1598_readvariableop_resource@
<model_438_batch_normalization_1598_readvariableop_1_resourceO
Kmodel_438_batch_normalization_1598_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_438_batch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource8
4model_438_conv2d_1608_conv2d_readvariableop_resource9
5model_438_conv2d_1608_biasadd_readvariableop_resource>
:model_438_batch_normalization_1608_readvariableop_resource@
<model_438_batch_normalization_1608_readvariableop_1_resourceO
Kmodel_438_batch_normalization_1608_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_438_batch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource8
4model_438_conv2d_1612_conv2d_readvariableop_resource9
5model_438_conv2d_1612_biasadd_readvariableop_resource>
:model_438_batch_normalization_1612_readvariableop_resource@
<model_438_batch_normalization_1612_readvariableop_1_resourceO
Kmodel_438_batch_normalization_1612_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_438_batch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource8
4model_438_conv2d_1625_conv2d_readvariableop_resource9
5model_438_conv2d_1625_biasadd_readvariableop_resource>
:model_438_batch_normalization_1625_readvariableop_resource@
<model_438_batch_normalization_1625_readvariableop_1_resourceO
Kmodel_438_batch_normalization_1625_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_438_batch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource6
2model_438_dense_878_matmul_readvariableop_resource7
3model_438_dense_878_biasadd_readvariableop_resource6
2model_438_dense_879_matmul_readvariableop_resource7
3model_438_dense_879_biasadd_readvariableop_resource
identity??@model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?Bmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?/model_438/batch_normalization_15/ReadVariableOp?1model_438/batch_normalization_15/ReadVariableOp_1?Bmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?Dmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?1model_438/batch_normalization_1592/ReadVariableOp?3model_438/batch_normalization_1592/ReadVariableOp_1?Bmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?Dmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?1model_438/batch_normalization_1598/ReadVariableOp?3model_438/batch_normalization_1598/ReadVariableOp_1?Bmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?Dmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?1model_438/batch_normalization_1608/ReadVariableOp?3model_438/batch_normalization_1608/ReadVariableOp_1?Bmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?Dmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?1model_438/batch_normalization_1612/ReadVariableOp?3model_438/batch_normalization_1612/ReadVariableOp_1?Bmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?Dmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?1model_438/batch_normalization_1625/ReadVariableOp?3model_438/batch_normalization_1625/ReadVariableOp_1?@model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Bmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?/model_438/batch_normalization_34/ReadVariableOp?1model_438/batch_normalization_34/ReadVariableOp_1??model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Amodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?.model_438/batch_normalization_4/ReadVariableOp?0model_438/batch_normalization_4/ReadVariableOp_1?*model_438/conv2d_15/BiasAdd/ReadVariableOp?)model_438/conv2d_15/Conv2D/ReadVariableOp?,model_438/conv2d_1592/BiasAdd/ReadVariableOp?+model_438/conv2d_1592/Conv2D/ReadVariableOp?,model_438/conv2d_1598/BiasAdd/ReadVariableOp?+model_438/conv2d_1598/Conv2D/ReadVariableOp?,model_438/conv2d_1608/BiasAdd/ReadVariableOp?+model_438/conv2d_1608/Conv2D/ReadVariableOp?,model_438/conv2d_1612/BiasAdd/ReadVariableOp?+model_438/conv2d_1612/Conv2D/ReadVariableOp?,model_438/conv2d_1625/BiasAdd/ReadVariableOp?+model_438/conv2d_1625/Conv2D/ReadVariableOp?*model_438/conv2d_34/BiasAdd/ReadVariableOp?)model_438/conv2d_34/Conv2D/ReadVariableOp?)model_438/conv2d_4/BiasAdd/ReadVariableOp?(model_438/conv2d_4/Conv2D/ReadVariableOp?*model_438/dense_878/BiasAdd/ReadVariableOp?)model_438/dense_878/MatMul/ReadVariableOp?*model_438/dense_879/BiasAdd/ReadVariableOp?)model_438/dense_879/MatMul/ReadVariableOp?
(model_438/conv2d_4/Conv2D/ReadVariableOpReadVariableOp1model_438_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_438/conv2d_4/Conv2D/ReadVariableOp?
model_438/conv2d_4/Conv2DConv2Dinput_10model_438/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_438/conv2d_4/Conv2D?
)model_438/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2model_438_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_438/conv2d_4/BiasAdd/ReadVariableOp?
model_438/conv2d_4/BiasAddBiasAdd"model_438/conv2d_4/Conv2D:output:01model_438/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_438/conv2d_4/BiasAdd?
.model_438/batch_normalization_4/ReadVariableOpReadVariableOp7model_438_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype020
.model_438/batch_normalization_4/ReadVariableOp?
0model_438/batch_normalization_4/ReadVariableOp_1ReadVariableOp9model_438_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_438/batch_normalization_4/ReadVariableOp_1?
?model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_438_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Amodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_438_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
0model_438/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#model_438/conv2d_4/BiasAdd:output:06model_438/batch_normalization_4/ReadVariableOp:value:08model_438/batch_normalization_4/ReadVariableOp_1:value:0Gmodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Imodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 22
0model_438/batch_normalization_4/FusedBatchNormV3?
model_438/re_lu_4/ReluRelu4model_438/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_438/re_lu_4/Relu?
!model_438/max_pooling2d_4/MaxPoolMaxPool$model_438/re_lu_4/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2#
!model_438/max_pooling2d_4/MaxPool?
)model_438/conv2d_15/Conv2D/ReadVariableOpReadVariableOp2model_438_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)model_438/conv2d_15/Conv2D/ReadVariableOp?
model_438/conv2d_15/Conv2DConv2D*model_438/max_pooling2d_4/MaxPool:output:01model_438/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_438/conv2d_15/Conv2D?
*model_438/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3model_438_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_438/conv2d_15/BiasAdd/ReadVariableOp?
model_438/conv2d_15/BiasAddBiasAdd#model_438/conv2d_15/Conv2D:output:02model_438/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_438/conv2d_15/BiasAdd?
/model_438/batch_normalization_15/ReadVariableOpReadVariableOp8model_438_batch_normalization_15_readvariableop_resource*
_output_shapes
:*
dtype021
/model_438/batch_normalization_15/ReadVariableOp?
1model_438/batch_normalization_15/ReadVariableOp_1ReadVariableOp:model_438_batch_normalization_15_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_438/batch_normalization_15/ReadVariableOp_1?
@model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_438_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
Bmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_438_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
1model_438/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3$model_438/conv2d_15/BiasAdd:output:07model_438/batch_normalization_15/ReadVariableOp:value:09model_438/batch_normalization_15/ReadVariableOp_1:value:0Hmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 23
1model_438/batch_normalization_15/FusedBatchNormV3?
model_438/re_lu_15/ReluRelu5model_438/batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_438/re_lu_15/Relu?
)model_438/conv2d_34/Conv2D/ReadVariableOpReadVariableOp2model_438_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)model_438/conv2d_34/Conv2D/ReadVariableOp?
model_438/conv2d_34/Conv2DConv2D%model_438/re_lu_15/Relu:activations:01model_438/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_438/conv2d_34/Conv2D?
*model_438/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp3model_438_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_438/conv2d_34/BiasAdd/ReadVariableOp?
model_438/conv2d_34/BiasAddBiasAdd#model_438/conv2d_34/Conv2D:output:02model_438/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_438/conv2d_34/BiasAdd?
/model_438/batch_normalization_34/ReadVariableOpReadVariableOp8model_438_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype021
/model_438/batch_normalization_34/ReadVariableOp?
1model_438/batch_normalization_34/ReadVariableOp_1ReadVariableOp:model_438_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_438/batch_normalization_34/ReadVariableOp_1?
@model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_438_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
Bmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_438_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
1model_438/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3$model_438/conv2d_34/BiasAdd:output:07model_438/batch_normalization_34/ReadVariableOp:value:09model_438/batch_normalization_34/ReadVariableOp_1:value:0Hmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 23
1model_438/batch_normalization_34/FusedBatchNormV3?
model_438/re_lu_34/ReluRelu5model_438/batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_438/re_lu_34/Relu?
"model_438/max_pooling2d_34/MaxPoolMaxPool%model_438/re_lu_34/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2$
"model_438/max_pooling2d_34/MaxPool?
+model_438/conv2d_1592/Conv2D/ReadVariableOpReadVariableOp4model_438_conv2d_1592_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+model_438/conv2d_1592/Conv2D/ReadVariableOp?
model_438/conv2d_1592/Conv2DConv2D+model_438/max_pooling2d_34/MaxPool:output:03model_438/conv2d_1592/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_438/conv2d_1592/Conv2D?
,model_438/conv2d_1592/BiasAdd/ReadVariableOpReadVariableOp5model_438_conv2d_1592_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_438/conv2d_1592/BiasAdd/ReadVariableOp?
model_438/conv2d_1592/BiasAddBiasAdd%model_438/conv2d_1592/Conv2D:output:04model_438/conv2d_1592/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_438/conv2d_1592/BiasAdd?
1model_438/batch_normalization_1592/ReadVariableOpReadVariableOp:model_438_batch_normalization_1592_readvariableop_resource*
_output_shapes
: *
dtype023
1model_438/batch_normalization_1592/ReadVariableOp?
3model_438/batch_normalization_1592/ReadVariableOp_1ReadVariableOp<model_438_batch_normalization_1592_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_438/batch_normalization_1592/ReadVariableOp_1?
Bmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_438_batch_normalization_1592_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?
Dmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_438_batch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?
3model_438/batch_normalization_1592/FusedBatchNormV3FusedBatchNormV3&model_438/conv2d_1592/BiasAdd:output:09model_438/batch_normalization_1592/ReadVariableOp:value:0;model_438/batch_normalization_1592/ReadVariableOp_1:value:0Jmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_438/batch_normalization_1592/FusedBatchNormV3?
model_438/re_lu_1484/ReluRelu7model_438/batch_normalization_1592/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_438/re_lu_1484/Relu?
$model_438/max_pooling2d_1484/MaxPoolMaxPool'model_438/re_lu_1484/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2&
$model_438/max_pooling2d_1484/MaxPool?
+model_438/conv2d_1598/Conv2D/ReadVariableOpReadVariableOp4model_438_conv2d_1598_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+model_438/conv2d_1598/Conv2D/ReadVariableOp?
model_438/conv2d_1598/Conv2DConv2D-model_438/max_pooling2d_1484/MaxPool:output:03model_438/conv2d_1598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_438/conv2d_1598/Conv2D?
,model_438/conv2d_1598/BiasAdd/ReadVariableOpReadVariableOp5model_438_conv2d_1598_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_438/conv2d_1598/BiasAdd/ReadVariableOp?
model_438/conv2d_1598/BiasAddBiasAdd%model_438/conv2d_1598/Conv2D:output:04model_438/conv2d_1598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_438/conv2d_1598/BiasAdd?
1model_438/batch_normalization_1598/ReadVariableOpReadVariableOp:model_438_batch_normalization_1598_readvariableop_resource*
_output_shapes
: *
dtype023
1model_438/batch_normalization_1598/ReadVariableOp?
3model_438/batch_normalization_1598/ReadVariableOp_1ReadVariableOp<model_438_batch_normalization_1598_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_438/batch_normalization_1598/ReadVariableOp_1?
Bmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_438_batch_normalization_1598_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?
Dmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_438_batch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?
3model_438/batch_normalization_1598/FusedBatchNormV3FusedBatchNormV3&model_438/conv2d_1598/BiasAdd:output:09model_438/batch_normalization_1598/ReadVariableOp:value:0;model_438/batch_normalization_1598/ReadVariableOp_1:value:0Jmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_438/batch_normalization_1598/FusedBatchNormV3?
model_438/re_lu_1490/ReluRelu7model_438/batch_normalization_1598/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_438/re_lu_1490/Relu?
$model_438/max_pooling2d_1490/MaxPoolMaxPool'model_438/re_lu_1490/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2&
$model_438/max_pooling2d_1490/MaxPool?
model_438/add_765/addAddV2-model_438/max_pooling2d_1484/MaxPool:output:0-model_438/max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
model_438/add_765/add?
+model_438/conv2d_1608/Conv2D/ReadVariableOpReadVariableOp4model_438_conv2d_1608_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+model_438/conv2d_1608/Conv2D/ReadVariableOp?
model_438/conv2d_1608/Conv2DConv2Dmodel_438/add_765/add:z:03model_438/conv2d_1608/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_438/conv2d_1608/Conv2D?
,model_438/conv2d_1608/BiasAdd/ReadVariableOpReadVariableOp5model_438_conv2d_1608_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_438/conv2d_1608/BiasAdd/ReadVariableOp?
model_438/conv2d_1608/BiasAddBiasAdd%model_438/conv2d_1608/Conv2D:output:04model_438/conv2d_1608/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_438/conv2d_1608/BiasAdd?
1model_438/batch_normalization_1608/ReadVariableOpReadVariableOp:model_438_batch_normalization_1608_readvariableop_resource*
_output_shapes
: *
dtype023
1model_438/batch_normalization_1608/ReadVariableOp?
3model_438/batch_normalization_1608/ReadVariableOp_1ReadVariableOp<model_438_batch_normalization_1608_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_438/batch_normalization_1608/ReadVariableOp_1?
Bmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_438_batch_normalization_1608_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?
Dmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_438_batch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?
3model_438/batch_normalization_1608/FusedBatchNormV3FusedBatchNormV3&model_438/conv2d_1608/BiasAdd:output:09model_438/batch_normalization_1608/ReadVariableOp:value:0;model_438/batch_normalization_1608/ReadVariableOp_1:value:0Jmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_438/batch_normalization_1608/FusedBatchNormV3?
model_438/re_lu_1500/ReluRelu7model_438/batch_normalization_1608/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_438/re_lu_1500/Relu?
$model_438/max_pooling2d_1500/MaxPoolMaxPool'model_438/re_lu_1500/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2&
$model_438/max_pooling2d_1500/MaxPool?
model_438/add_766/addAddV2-model_438/max_pooling2d_1500/MaxPool:output:0-model_438/max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
model_438/add_766/add?
+model_438/conv2d_1612/Conv2D/ReadVariableOpReadVariableOp4model_438_conv2d_1612_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+model_438/conv2d_1612/Conv2D/ReadVariableOp?
model_438/conv2d_1612/Conv2DConv2Dmodel_438/add_766/add:z:03model_438/conv2d_1612/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_438/conv2d_1612/Conv2D?
,model_438/conv2d_1612/BiasAdd/ReadVariableOpReadVariableOp5model_438_conv2d_1612_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_438/conv2d_1612/BiasAdd/ReadVariableOp?
model_438/conv2d_1612/BiasAddBiasAdd%model_438/conv2d_1612/Conv2D:output:04model_438/conv2d_1612/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_438/conv2d_1612/BiasAdd?
1model_438/batch_normalization_1612/ReadVariableOpReadVariableOp:model_438_batch_normalization_1612_readvariableop_resource*
_output_shapes
: *
dtype023
1model_438/batch_normalization_1612/ReadVariableOp?
3model_438/batch_normalization_1612/ReadVariableOp_1ReadVariableOp<model_438_batch_normalization_1612_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_438/batch_normalization_1612/ReadVariableOp_1?
Bmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_438_batch_normalization_1612_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?
Dmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_438_batch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?
3model_438/batch_normalization_1612/FusedBatchNormV3FusedBatchNormV3&model_438/conv2d_1612/BiasAdd:output:09model_438/batch_normalization_1612/ReadVariableOp:value:0;model_438/batch_normalization_1612/ReadVariableOp_1:value:0Jmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_438/batch_normalization_1612/FusedBatchNormV3?
model_438/re_lu_1504/ReluRelu7model_438/batch_normalization_1612/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_438/re_lu_1504/Relu?
$model_438/max_pooling2d_1504/MaxPoolMaxPool'model_438/re_lu_1504/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2&
$model_438/max_pooling2d_1504/MaxPool?
model_438/add_767/addAddV2-model_438/max_pooling2d_1504/MaxPool:output:0-model_438/max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
model_438/add_767/add?
+model_438/conv2d_1625/Conv2D/ReadVariableOpReadVariableOp4model_438_conv2d_1625_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+model_438/conv2d_1625/Conv2D/ReadVariableOp?
model_438/conv2d_1625/Conv2DConv2Dmodel_438/add_767/add:z:03model_438/conv2d_1625/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_438/conv2d_1625/Conv2D?
,model_438/conv2d_1625/BiasAdd/ReadVariableOpReadVariableOp5model_438_conv2d_1625_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_438/conv2d_1625/BiasAdd/ReadVariableOp?
model_438/conv2d_1625/BiasAddBiasAdd%model_438/conv2d_1625/Conv2D:output:04model_438/conv2d_1625/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_438/conv2d_1625/BiasAdd?
1model_438/batch_normalization_1625/ReadVariableOpReadVariableOp:model_438_batch_normalization_1625_readvariableop_resource*
_output_shapes
: *
dtype023
1model_438/batch_normalization_1625/ReadVariableOp?
3model_438/batch_normalization_1625/ReadVariableOp_1ReadVariableOp<model_438_batch_normalization_1625_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_438/batch_normalization_1625/ReadVariableOp_1?
Bmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_438_batch_normalization_1625_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?
Dmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_438_batch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?
3model_438/batch_normalization_1625/FusedBatchNormV3FusedBatchNormV3&model_438/conv2d_1625/BiasAdd:output:09model_438/batch_normalization_1625/ReadVariableOp:value:0;model_438/batch_normalization_1625/ReadVariableOp_1:value:0Jmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_438/batch_normalization_1625/FusedBatchNormV3?
model_438/re_lu_1517/ReluRelu7model_438/batch_normalization_1625/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_438/re_lu_1517/Relu?
model_438/flatten_439/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
model_438/flatten_439/Const?
model_438/flatten_439/ReshapeReshape'model_438/re_lu_1517/Relu:activations:0$model_438/flatten_439/Const:output:0*
T0*)
_output_shapes
:???????????2
model_438/flatten_439/Reshape?
)model_438/dense_878/MatMul/ReadVariableOpReadVariableOp2model_438_dense_878_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02+
)model_438/dense_878/MatMul/ReadVariableOp?
model_438/dense_878/MatMulMatMul&model_438/flatten_439/Reshape:output:01model_438/dense_878/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_438/dense_878/MatMul?
*model_438/dense_878/BiasAdd/ReadVariableOpReadVariableOp3model_438_dense_878_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_438/dense_878/BiasAdd/ReadVariableOp?
model_438/dense_878/BiasAddBiasAdd$model_438/dense_878/MatMul:product:02model_438/dense_878/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_438/dense_878/BiasAdd?
model_438/dense_878/ReluRelu$model_438/dense_878/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_438/dense_878/Relu?
)model_438/dense_879/MatMul/ReadVariableOpReadVariableOp2model_438_dense_879_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02+
)model_438/dense_879/MatMul/ReadVariableOp?
model_438/dense_879/MatMulMatMul&model_438/dense_878/Relu:activations:01model_438/dense_879/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_438/dense_879/MatMul?
*model_438/dense_879/BiasAdd/ReadVariableOpReadVariableOp3model_438_dense_879_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*model_438/dense_879/BiasAdd/ReadVariableOp?
model_438/dense_879/BiasAddBiasAdd$model_438/dense_879/MatMul:product:02model_438/dense_879/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_438/dense_879/BiasAdd?
model_438/dense_879/SoftmaxSoftmax$model_438/dense_879/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model_438/dense_879/Softmax?
IdentityIdentity%model_438/dense_879/Softmax:softmax:0A^model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOpC^model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_10^model_438/batch_normalization_15/ReadVariableOp2^model_438/batch_normalization_15/ReadVariableOp_1C^model_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOpE^model_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_12^model_438/batch_normalization_1592/ReadVariableOp4^model_438/batch_normalization_1592/ReadVariableOp_1C^model_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOpE^model_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_12^model_438/batch_normalization_1598/ReadVariableOp4^model_438/batch_normalization_1598/ReadVariableOp_1C^model_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOpE^model_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_12^model_438/batch_normalization_1608/ReadVariableOp4^model_438/batch_normalization_1608/ReadVariableOp_1C^model_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOpE^model_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_12^model_438/batch_normalization_1612/ReadVariableOp4^model_438/batch_normalization_1612/ReadVariableOp_1C^model_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOpE^model_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_12^model_438/batch_normalization_1625/ReadVariableOp4^model_438/batch_normalization_1625/ReadVariableOp_1A^model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOpC^model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_10^model_438/batch_normalization_34/ReadVariableOp2^model_438/batch_normalization_34/ReadVariableOp_1@^model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOpB^model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/^model_438/batch_normalization_4/ReadVariableOp1^model_438/batch_normalization_4/ReadVariableOp_1+^model_438/conv2d_15/BiasAdd/ReadVariableOp*^model_438/conv2d_15/Conv2D/ReadVariableOp-^model_438/conv2d_1592/BiasAdd/ReadVariableOp,^model_438/conv2d_1592/Conv2D/ReadVariableOp-^model_438/conv2d_1598/BiasAdd/ReadVariableOp,^model_438/conv2d_1598/Conv2D/ReadVariableOp-^model_438/conv2d_1608/BiasAdd/ReadVariableOp,^model_438/conv2d_1608/Conv2D/ReadVariableOp-^model_438/conv2d_1612/BiasAdd/ReadVariableOp,^model_438/conv2d_1612/Conv2D/ReadVariableOp-^model_438/conv2d_1625/BiasAdd/ReadVariableOp,^model_438/conv2d_1625/Conv2D/ReadVariableOp+^model_438/conv2d_34/BiasAdd/ReadVariableOp*^model_438/conv2d_34/Conv2D/ReadVariableOp*^model_438/conv2d_4/BiasAdd/ReadVariableOp)^model_438/conv2d_4/Conv2D/ReadVariableOp+^model_438/dense_878/BiasAdd/ReadVariableOp*^model_438/dense_878/MatMul/ReadVariableOp+^model_438/dense_879/BiasAdd/ReadVariableOp*^model_438/dense_879/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2?
@model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp@model_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
Bmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Bmodel_438/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12b
/model_438/batch_normalization_15/ReadVariableOp/model_438/batch_normalization_15/ReadVariableOp2f
1model_438/batch_normalization_15/ReadVariableOp_11model_438/batch_normalization_15/ReadVariableOp_12?
Bmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOpBmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp2?
Dmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1Dmodel_438/batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_12f
1model_438/batch_normalization_1592/ReadVariableOp1model_438/batch_normalization_1592/ReadVariableOp2j
3model_438/batch_normalization_1592/ReadVariableOp_13model_438/batch_normalization_1592/ReadVariableOp_12?
Bmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOpBmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp2?
Dmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1Dmodel_438/batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_12f
1model_438/batch_normalization_1598/ReadVariableOp1model_438/batch_normalization_1598/ReadVariableOp2j
3model_438/batch_normalization_1598/ReadVariableOp_13model_438/batch_normalization_1598/ReadVariableOp_12?
Bmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOpBmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp2?
Dmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1Dmodel_438/batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_12f
1model_438/batch_normalization_1608/ReadVariableOp1model_438/batch_normalization_1608/ReadVariableOp2j
3model_438/batch_normalization_1608/ReadVariableOp_13model_438/batch_normalization_1608/ReadVariableOp_12?
Bmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOpBmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp2?
Dmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1Dmodel_438/batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_12f
1model_438/batch_normalization_1612/ReadVariableOp1model_438/batch_normalization_1612/ReadVariableOp2j
3model_438/batch_normalization_1612/ReadVariableOp_13model_438/batch_normalization_1612/ReadVariableOp_12?
Bmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOpBmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp2?
Dmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1Dmodel_438/batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_12f
1model_438/batch_normalization_1625/ReadVariableOp1model_438/batch_normalization_1625/ReadVariableOp2j
3model_438/batch_normalization_1625/ReadVariableOp_13model_438/batch_normalization_1625/ReadVariableOp_12?
@model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp@model_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Bmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Bmodel_438/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12b
/model_438/batch_normalization_34/ReadVariableOp/model_438/batch_normalization_34/ReadVariableOp2f
1model_438/batch_normalization_34/ReadVariableOp_11model_438/batch_normalization_34/ReadVariableOp_12?
?model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?model_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Amodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Amodel_438/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12`
.model_438/batch_normalization_4/ReadVariableOp.model_438/batch_normalization_4/ReadVariableOp2d
0model_438/batch_normalization_4/ReadVariableOp_10model_438/batch_normalization_4/ReadVariableOp_12X
*model_438/conv2d_15/BiasAdd/ReadVariableOp*model_438/conv2d_15/BiasAdd/ReadVariableOp2V
)model_438/conv2d_15/Conv2D/ReadVariableOp)model_438/conv2d_15/Conv2D/ReadVariableOp2\
,model_438/conv2d_1592/BiasAdd/ReadVariableOp,model_438/conv2d_1592/BiasAdd/ReadVariableOp2Z
+model_438/conv2d_1592/Conv2D/ReadVariableOp+model_438/conv2d_1592/Conv2D/ReadVariableOp2\
,model_438/conv2d_1598/BiasAdd/ReadVariableOp,model_438/conv2d_1598/BiasAdd/ReadVariableOp2Z
+model_438/conv2d_1598/Conv2D/ReadVariableOp+model_438/conv2d_1598/Conv2D/ReadVariableOp2\
,model_438/conv2d_1608/BiasAdd/ReadVariableOp,model_438/conv2d_1608/BiasAdd/ReadVariableOp2Z
+model_438/conv2d_1608/Conv2D/ReadVariableOp+model_438/conv2d_1608/Conv2D/ReadVariableOp2\
,model_438/conv2d_1612/BiasAdd/ReadVariableOp,model_438/conv2d_1612/BiasAdd/ReadVariableOp2Z
+model_438/conv2d_1612/Conv2D/ReadVariableOp+model_438/conv2d_1612/Conv2D/ReadVariableOp2\
,model_438/conv2d_1625/BiasAdd/ReadVariableOp,model_438/conv2d_1625/BiasAdd/ReadVariableOp2Z
+model_438/conv2d_1625/Conv2D/ReadVariableOp+model_438/conv2d_1625/Conv2D/ReadVariableOp2X
*model_438/conv2d_34/BiasAdd/ReadVariableOp*model_438/conv2d_34/BiasAdd/ReadVariableOp2V
)model_438/conv2d_34/Conv2D/ReadVariableOp)model_438/conv2d_34/Conv2D/ReadVariableOp2V
)model_438/conv2d_4/BiasAdd/ReadVariableOp)model_438/conv2d_4/BiasAdd/ReadVariableOp2T
(model_438/conv2d_4/Conv2D/ReadVariableOp(model_438/conv2d_4/Conv2D/ReadVariableOp2X
*model_438/dense_878/BiasAdd/ReadVariableOp*model_438/dense_878/BiasAdd/ReadVariableOp2V
)model_438/dense_878/MatMul/ReadVariableOp)model_438/dense_878/MatMul/ReadVariableOp2X
*model_438/dense_879/BiasAdd/ReadVariableOp*model_438/dense_879/BiasAdd/ReadVariableOp2V
)model_438/dense_879/MatMul/ReadVariableOp)model_438/dense_879/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
F__inference_dense_879_layer_call_and_return_conditional_losses_5507952

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5506907

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_4_layer_call_fn_5509499

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55061072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_re_lu_1484_layer_call_fn_5509993

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_55073952
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_5510510

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_15_layer_call_fn_5509656

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_55062232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_15_layer_call_and_return_conditional_losses_5507170

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
+__inference_conv2d_34_layer_call_fn_5509698

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_55071882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1598_layer_call_fn_5510012

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_55074142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
??
?
F__inference_model_438_layer_call_and_return_conditional_losses_5507969
input_1
conv2d_4_5506974
conv2d_4_5506976!
batch_normalization_4_5507043!
batch_normalization_4_5507045!
batch_normalization_4_5507047!
batch_normalization_4_5507049
conv2d_15_5507087
conv2d_15_5507089"
batch_normalization_15_5507156"
batch_normalization_15_5507158"
batch_normalization_15_5507160"
batch_normalization_15_5507162
conv2d_34_5507199
conv2d_34_5507201"
batch_normalization_34_5507268"
batch_normalization_34_5507270"
batch_normalization_34_5507272"
batch_normalization_34_5507274
conv2d_1592_5507312
conv2d_1592_5507314$
 batch_normalization_1592_5507381$
 batch_normalization_1592_5507383$
 batch_normalization_1592_5507385$
 batch_normalization_1592_5507387
conv2d_1598_5507425
conv2d_1598_5507427$
 batch_normalization_1598_5507494$
 batch_normalization_1598_5507496$
 batch_normalization_1598_5507498$
 batch_normalization_1598_5507500
conv2d_1608_5507553
conv2d_1608_5507555$
 batch_normalization_1608_5507622$
 batch_normalization_1608_5507624$
 batch_normalization_1608_5507626$
 batch_normalization_1608_5507628
conv2d_1612_5507681
conv2d_1612_5507683$
 batch_normalization_1612_5507750$
 batch_normalization_1612_5507752$
 batch_normalization_1612_5507754$
 batch_normalization_1612_5507756
conv2d_1625_5507809
conv2d_1625_5507811$
 batch_normalization_1625_5507878$
 batch_normalization_1625_5507880$
 batch_normalization_1625_5507882$
 batch_normalization_1625_5507884
dense_878_5507936
dense_878_5507938
dense_879_5507963
dense_879_5507965
identity??.batch_normalization_15/StatefulPartitionedCall?0batch_normalization_1592/StatefulPartitionedCall?0batch_normalization_1598/StatefulPartitionedCall?0batch_normalization_1608/StatefulPartitionedCall?0batch_normalization_1612/StatefulPartitionedCall?0batch_normalization_1625/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#conv2d_1592/StatefulPartitionedCall?#conv2d_1598/StatefulPartitionedCall?#conv2d_1608/StatefulPartitionedCall?#conv2d_1612/StatefulPartitionedCall?#conv2d_1625/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?!dense_878/StatefulPartitionedCall?!dense_879/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_4_5506974conv2d_4_5506976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_55069632"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_5507043batch_normalization_4_5507045batch_normalization_4_5507047batch_normalization_4_5507049*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55069982/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_55070572
re_lu_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_55061552!
max_pooling2d_4/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_15_5507087conv2d_15_5507089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_55070762#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_5507156batch_normalization_15_5507158batch_normalization_15_5507160batch_normalization_15_5507162*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_550711120
.batch_normalization_15/StatefulPartitionedCall?
re_lu_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_15_layer_call_and_return_conditional_losses_55071702
re_lu_15/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_15/PartitionedCall:output:0conv2d_34_5507199conv2d_34_5507201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_55071882#
!conv2d_34/StatefulPartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_34_5507268batch_normalization_34_5507270batch_normalization_34_5507272batch_normalization_34_5507274*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_550722320
.batch_normalization_34/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_34_layer_call_and_return_conditional_losses_55072822
re_lu_34/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_55063752"
 max_pooling2d_34/PartitionedCall?
#conv2d_1592/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_1592_5507312conv2d_1592_5507314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_55073012%
#conv2d_1592/StatefulPartitionedCall?
0batch_normalization_1592/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1592/StatefulPartitionedCall:output:0 batch_normalization_1592_5507381 batch_normalization_1592_5507383 batch_normalization_1592_5507385 batch_normalization_1592_5507387*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_550733622
0batch_normalization_1592/StatefulPartitionedCall?
re_lu_1484/PartitionedCallPartitionedCall9batch_normalization_1592/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_55073952
re_lu_1484/PartitionedCall?
"max_pooling2d_1484/PartitionedCallPartitionedCall#re_lu_1484/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_55064912$
"max_pooling2d_1484/PartitionedCall?
#conv2d_1598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0conv2d_1598_5507425conv2d_1598_5507427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_55074142%
#conv2d_1598/StatefulPartitionedCall?
0batch_normalization_1598/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1598/StatefulPartitionedCall:output:0 batch_normalization_1598_5507494 batch_normalization_1598_5507496 batch_normalization_1598_5507498 batch_normalization_1598_5507500*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_550744922
0batch_normalization_1598/StatefulPartitionedCall?
re_lu_1490/PartitionedCallPartitionedCall9batch_normalization_1598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_55075082
re_lu_1490/PartitionedCall?
"max_pooling2d_1490/PartitionedCallPartitionedCall#re_lu_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_55066072$
"max_pooling2d_1490/PartitionedCall?
add_765/PartitionedCallPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_765_layer_call_and_return_conditional_losses_55075232
add_765/PartitionedCall?
#conv2d_1608/StatefulPartitionedCallStatefulPartitionedCall add_765/PartitionedCall:output:0conv2d_1608_5507553conv2d_1608_5507555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_55075422%
#conv2d_1608/StatefulPartitionedCall?
0batch_normalization_1608/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1608/StatefulPartitionedCall:output:0 batch_normalization_1608_5507622 batch_normalization_1608_5507624 batch_normalization_1608_5507626 batch_normalization_1608_5507628*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_550757722
0batch_normalization_1608/StatefulPartitionedCall?
re_lu_1500/PartitionedCallPartitionedCall9batch_normalization_1608/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_55076362
re_lu_1500/PartitionedCall?
"max_pooling2d_1500/PartitionedCallPartitionedCall#re_lu_1500/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_55067232$
"max_pooling2d_1500/PartitionedCall?
add_766/PartitionedCallPartitionedCall+max_pooling2d_1500/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_766_layer_call_and_return_conditional_losses_55076512
add_766/PartitionedCall?
#conv2d_1612/StatefulPartitionedCallStatefulPartitionedCall add_766/PartitionedCall:output:0conv2d_1612_5507681conv2d_1612_5507683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_55076702%
#conv2d_1612/StatefulPartitionedCall?
0batch_normalization_1612/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1612/StatefulPartitionedCall:output:0 batch_normalization_1612_5507750 batch_normalization_1612_5507752 batch_normalization_1612_5507754 batch_normalization_1612_5507756*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_550770522
0batch_normalization_1612/StatefulPartitionedCall?
re_lu_1504/PartitionedCallPartitionedCall9batch_normalization_1612/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_55077642
re_lu_1504/PartitionedCall?
"max_pooling2d_1504/PartitionedCallPartitionedCall#re_lu_1504/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_55068392$
"max_pooling2d_1504/PartitionedCall?
add_767/PartitionedCallPartitionedCall+max_pooling2d_1504/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_767_layer_call_and_return_conditional_losses_55077792
add_767/PartitionedCall?
#conv2d_1625/StatefulPartitionedCallStatefulPartitionedCall add_767/PartitionedCall:output:0conv2d_1625_5507809conv2d_1625_5507811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_55077982%
#conv2d_1625/StatefulPartitionedCall?
0batch_normalization_1625/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1625/StatefulPartitionedCall:output:0 batch_normalization_1625_5507878 batch_normalization_1625_5507880 batch_normalization_1625_5507882 batch_normalization_1625_5507884*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_550783322
0batch_normalization_1625/StatefulPartitionedCall?
re_lu_1517/PartitionedCallPartitionedCall9batch_normalization_1625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_55078922
re_lu_1517/PartitionedCall?
flatten_439/PartitionedCallPartitionedCall#re_lu_1517/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_439_layer_call_and_return_conditional_losses_55079062
flatten_439/PartitionedCall?
!dense_878/StatefulPartitionedCallStatefulPartitionedCall$flatten_439/PartitionedCall:output:0dense_878_5507936dense_878_5507938*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_55079252#
!dense_878/StatefulPartitionedCall?
!dense_879/StatefulPartitionedCallStatefulPartitionedCall*dense_878/StatefulPartitionedCall:output:0dense_879_5507963dense_879_5507965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_879_layer_call_and_return_conditional_losses_55079522#
!dense_879/StatefulPartitionedCall?
IdentityIdentity*dense_879/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall1^batch_normalization_1592/StatefulPartitionedCall1^batch_normalization_1598/StatefulPartitionedCall1^batch_normalization_1608/StatefulPartitionedCall1^batch_normalization_1612/StatefulPartitionedCall1^batch_normalization_1625/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^conv2d_1592/StatefulPartitionedCall$^conv2d_1598/StatefulPartitionedCall$^conv2d_1608/StatefulPartitionedCall$^conv2d_1612/StatefulPartitionedCall$^conv2d_1625/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall"^dense_879/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2d
0batch_normalization_1592/StatefulPartitionedCall0batch_normalization_1592/StatefulPartitionedCall2d
0batch_normalization_1598/StatefulPartitionedCall0batch_normalization_1598/StatefulPartitionedCall2d
0batch_normalization_1608/StatefulPartitionedCall0batch_normalization_1608/StatefulPartitionedCall2d
0batch_normalization_1612/StatefulPartitionedCall0batch_normalization_1612/StatefulPartitionedCall2d
0batch_normalization_1625/StatefulPartitionedCall0batch_normalization_1625/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#conv2d_1592/StatefulPartitionedCall#conv2d_1592/StatefulPartitionedCall2J
#conv2d_1598/StatefulPartitionedCall#conv2d_1598/StatefulPartitionedCall2J
#conv2d_1608/StatefulPartitionedCall#conv2d_1608/StatefulPartitionedCall2J
#conv2d_1612/StatefulPartitionedCall#conv2d_1612/StatefulPartitionedCall2J
#conv2d_1625/StatefulPartitionedCall#conv2d_1625/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall2F
!dense_879/StatefulPartitionedCall!dense_879/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5506155

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
p
D__inference_add_765_layer_call_and_return_conditional_losses_5510156
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?	
?
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_5507542

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510114

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
P
4__inference_max_pooling2d_1484_layer_call_fn_5506497

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_55064912
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509468

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_5506607

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5507723

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
a
E__inference_re_lu_34_layer_call_and_return_conditional_losses_5507282

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
+__inference_dense_879_layer_call_fn_5510708

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_879_layer_call_and_return_conditional_losses_55079522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_5507395

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1592_layer_call_fn_5509919

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_55064742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5507129

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509736

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
E
)__inference_re_lu_4_layer_call_fn_5509522

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_55070572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_5510314

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
??
?/
F__inference_model_438_layer_call_and_return_conditional_losses_5508950

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource2
.batch_normalization_34_readvariableop_resource4
0batch_normalization_34_readvariableop_1_resourceC
?batch_normalization_34_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1592_conv2d_readvariableop_resource/
+conv2d_1592_biasadd_readvariableop_resource4
0batch_normalization_1592_readvariableop_resource6
2batch_normalization_1592_readvariableop_1_resourceE
Abatch_normalization_1592_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1598_conv2d_readvariableop_resource/
+conv2d_1598_biasadd_readvariableop_resource4
0batch_normalization_1598_readvariableop_resource6
2batch_normalization_1598_readvariableop_1_resourceE
Abatch_normalization_1598_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1608_conv2d_readvariableop_resource/
+conv2d_1608_biasadd_readvariableop_resource4
0batch_normalization_1608_readvariableop_resource6
2batch_normalization_1608_readvariableop_1_resourceE
Abatch_normalization_1608_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1612_conv2d_readvariableop_resource/
+conv2d_1612_biasadd_readvariableop_resource4
0batch_normalization_1612_readvariableop_resource6
2batch_normalization_1612_readvariableop_1_resourceE
Abatch_normalization_1612_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1625_conv2d_readvariableop_resource/
+conv2d_1625_biasadd_readvariableop_resource4
0batch_normalization_1625_readvariableop_resource6
2batch_normalization_1625_readvariableop_1_resourceE
Abatch_normalization_1625_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource,
(dense_878_matmul_readvariableop_resource-
)dense_878_biasadd_readvariableop_resource,
(dense_879_matmul_readvariableop_resource-
)dense_879_biasadd_readvariableop_resource
identity??%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?'batch_normalization_1592/AssignNewValue?)batch_normalization_1592/AssignNewValue_1?8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1592/ReadVariableOp?)batch_normalization_1592/ReadVariableOp_1?'batch_normalization_1598/AssignNewValue?)batch_normalization_1598/AssignNewValue_1?8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1598/ReadVariableOp?)batch_normalization_1598/ReadVariableOp_1?'batch_normalization_1608/AssignNewValue?)batch_normalization_1608/AssignNewValue_1?8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1608/ReadVariableOp?)batch_normalization_1608/ReadVariableOp_1?'batch_normalization_1612/AssignNewValue?)batch_normalization_1612/AssignNewValue_1?8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1612/ReadVariableOp?)batch_normalization_1612/ReadVariableOp_1?'batch_normalization_1625/AssignNewValue?)batch_normalization_1625/AssignNewValue_1?8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1625/ReadVariableOp?)batch_normalization_1625/ReadVariableOp_1?%batch_normalization_34/AssignNewValue?'batch_normalization_34/AssignNewValue_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?"conv2d_1592/BiasAdd/ReadVariableOp?!conv2d_1592/Conv2D/ReadVariableOp?"conv2d_1598/BiasAdd/ReadVariableOp?!conv2d_1598/Conv2D/ReadVariableOp?"conv2d_1608/BiasAdd/ReadVariableOp?!conv2d_1608/Conv2D/ReadVariableOp?"conv2d_1612/BiasAdd/ReadVariableOp?!conv2d_1612/Conv2D/ReadVariableOp?"conv2d_1625/BiasAdd/ReadVariableOp?!conv2d_1625/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp? dense_878/BiasAdd/ReadVariableOp?dense_878/MatMul/ReadVariableOp? dense_879/BiasAdd/ReadVariableOp?dense_879/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_4/Relu?
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_15/BiasAdd?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
re_lu_15/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_15/Relu?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2Dre_lu_15/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_34/BiasAdd?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_34/ReadVariableOp?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_34/ReadVariableOp_1?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_34/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_34/FusedBatchNormV3?
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_34/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_34/AssignNewValue?
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_34/AssignNewValue_1?
re_lu_34/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_34/Relu?
max_pooling2d_34/MaxPoolMaxPoolre_lu_34/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_34/MaxPool?
!conv2d_1592/Conv2D/ReadVariableOpReadVariableOp*conv2d_1592_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_1592/Conv2D/ReadVariableOp?
conv2d_1592/Conv2DConv2D!max_pooling2d_34/MaxPool:output:0)conv2d_1592/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1592/Conv2D?
"conv2d_1592/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1592_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1592/BiasAdd/ReadVariableOp?
conv2d_1592/BiasAddBiasAddconv2d_1592/Conv2D:output:0*conv2d_1592/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1592/BiasAdd?
'batch_normalization_1592/ReadVariableOpReadVariableOp0batch_normalization_1592_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1592/ReadVariableOp?
)batch_normalization_1592/ReadVariableOp_1ReadVariableOp2batch_normalization_1592_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1592/ReadVariableOp_1?
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1592_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1592/FusedBatchNormV3FusedBatchNormV3conv2d_1592/BiasAdd:output:0/batch_normalization_1592/ReadVariableOp:value:01batch_normalization_1592/ReadVariableOp_1:value:0@batch_normalization_1592/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1592/FusedBatchNormV3?
'batch_normalization_1592/AssignNewValueAssignVariableOpAbatch_normalization_1592_fusedbatchnormv3_readvariableop_resource6batch_normalization_1592/FusedBatchNormV3:batch_mean:09^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1592/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1592/AssignNewValue?
)batch_normalization_1592/AssignNewValue_1AssignVariableOpCbatch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1592/FusedBatchNormV3:batch_variance:0;^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1592/AssignNewValue_1?
re_lu_1484/ReluRelu-batch_normalization_1592/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1484/Relu?
max_pooling2d_1484/MaxPoolMaxPoolre_lu_1484/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1484/MaxPool?
!conv2d_1598/Conv2D/ReadVariableOpReadVariableOp*conv2d_1598_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1598/Conv2D/ReadVariableOp?
conv2d_1598/Conv2DConv2D#max_pooling2d_1484/MaxPool:output:0)conv2d_1598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1598/Conv2D?
"conv2d_1598/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1598_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1598/BiasAdd/ReadVariableOp?
conv2d_1598/BiasAddBiasAddconv2d_1598/Conv2D:output:0*conv2d_1598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1598/BiasAdd?
'batch_normalization_1598/ReadVariableOpReadVariableOp0batch_normalization_1598_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1598/ReadVariableOp?
)batch_normalization_1598/ReadVariableOp_1ReadVariableOp2batch_normalization_1598_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1598/ReadVariableOp_1?
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1598_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1598/FusedBatchNormV3FusedBatchNormV3conv2d_1598/BiasAdd:output:0/batch_normalization_1598/ReadVariableOp:value:01batch_normalization_1598/ReadVariableOp_1:value:0@batch_normalization_1598/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1598/FusedBatchNormV3?
'batch_normalization_1598/AssignNewValueAssignVariableOpAbatch_normalization_1598_fusedbatchnormv3_readvariableop_resource6batch_normalization_1598/FusedBatchNormV3:batch_mean:09^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1598/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1598/AssignNewValue?
)batch_normalization_1598/AssignNewValue_1AssignVariableOpCbatch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1598/FusedBatchNormV3:batch_variance:0;^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1598/AssignNewValue_1?
re_lu_1490/ReluRelu-batch_normalization_1598/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1490/Relu?
max_pooling2d_1490/MaxPoolMaxPoolre_lu_1490/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1490/MaxPool?
add_765/addAddV2#max_pooling2d_1484/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_765/add?
!conv2d_1608/Conv2D/ReadVariableOpReadVariableOp*conv2d_1608_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1608/Conv2D/ReadVariableOp?
conv2d_1608/Conv2DConv2Dadd_765/add:z:0)conv2d_1608/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1608/Conv2D?
"conv2d_1608/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1608_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1608/BiasAdd/ReadVariableOp?
conv2d_1608/BiasAddBiasAddconv2d_1608/Conv2D:output:0*conv2d_1608/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1608/BiasAdd?
'batch_normalization_1608/ReadVariableOpReadVariableOp0batch_normalization_1608_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1608/ReadVariableOp?
)batch_normalization_1608/ReadVariableOp_1ReadVariableOp2batch_normalization_1608_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1608/ReadVariableOp_1?
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1608_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1608/FusedBatchNormV3FusedBatchNormV3conv2d_1608/BiasAdd:output:0/batch_normalization_1608/ReadVariableOp:value:01batch_normalization_1608/ReadVariableOp_1:value:0@batch_normalization_1608/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1608/FusedBatchNormV3?
'batch_normalization_1608/AssignNewValueAssignVariableOpAbatch_normalization_1608_fusedbatchnormv3_readvariableop_resource6batch_normalization_1608/FusedBatchNormV3:batch_mean:09^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1608/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1608/AssignNewValue?
)batch_normalization_1608/AssignNewValue_1AssignVariableOpCbatch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1608/FusedBatchNormV3:batch_variance:0;^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1608/AssignNewValue_1?
re_lu_1500/ReluRelu-batch_normalization_1608/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1500/Relu?
max_pooling2d_1500/MaxPoolMaxPoolre_lu_1500/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1500/MaxPool?
add_766/addAddV2#max_pooling2d_1500/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_766/add?
!conv2d_1612/Conv2D/ReadVariableOpReadVariableOp*conv2d_1612_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1612/Conv2D/ReadVariableOp?
conv2d_1612/Conv2DConv2Dadd_766/add:z:0)conv2d_1612/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1612/Conv2D?
"conv2d_1612/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1612_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1612/BiasAdd/ReadVariableOp?
conv2d_1612/BiasAddBiasAddconv2d_1612/Conv2D:output:0*conv2d_1612/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1612/BiasAdd?
'batch_normalization_1612/ReadVariableOpReadVariableOp0batch_normalization_1612_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1612/ReadVariableOp?
)batch_normalization_1612/ReadVariableOp_1ReadVariableOp2batch_normalization_1612_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1612/ReadVariableOp_1?
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1612_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1612/FusedBatchNormV3FusedBatchNormV3conv2d_1612/BiasAdd:output:0/batch_normalization_1612/ReadVariableOp:value:01batch_normalization_1612/ReadVariableOp_1:value:0@batch_normalization_1612/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1612/FusedBatchNormV3?
'batch_normalization_1612/AssignNewValueAssignVariableOpAbatch_normalization_1612_fusedbatchnormv3_readvariableop_resource6batch_normalization_1612/FusedBatchNormV3:batch_mean:09^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1612/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1612/AssignNewValue?
)batch_normalization_1612/AssignNewValue_1AssignVariableOpCbatch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1612/FusedBatchNormV3:batch_variance:0;^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1612/AssignNewValue_1?
re_lu_1504/ReluRelu-batch_normalization_1612/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1504/Relu?
max_pooling2d_1504/MaxPoolMaxPoolre_lu_1504/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1504/MaxPool?
add_767/addAddV2#max_pooling2d_1504/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_767/add?
!conv2d_1625/Conv2D/ReadVariableOpReadVariableOp*conv2d_1625_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1625/Conv2D/ReadVariableOp?
conv2d_1625/Conv2DConv2Dadd_767/add:z:0)conv2d_1625/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1625/Conv2D?
"conv2d_1625/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1625_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1625/BiasAdd/ReadVariableOp?
conv2d_1625/BiasAddBiasAddconv2d_1625/Conv2D:output:0*conv2d_1625/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1625/BiasAdd?
'batch_normalization_1625/ReadVariableOpReadVariableOp0batch_normalization_1625_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1625/ReadVariableOp?
)batch_normalization_1625/ReadVariableOp_1ReadVariableOp2batch_normalization_1625_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1625/ReadVariableOp_1?
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1625_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1625/FusedBatchNormV3FusedBatchNormV3conv2d_1625/BiasAdd:output:0/batch_normalization_1625/ReadVariableOp:value:01batch_normalization_1625/ReadVariableOp_1:value:0@batch_normalization_1625/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1625/FusedBatchNormV3?
'batch_normalization_1625/AssignNewValueAssignVariableOpAbatch_normalization_1625_fusedbatchnormv3_readvariableop_resource6batch_normalization_1625/FusedBatchNormV3:batch_mean:09^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1625/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1625/AssignNewValue?
)batch_normalization_1625/AssignNewValue_1AssignVariableOpCbatch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1625/FusedBatchNormV3:batch_variance:0;^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1625/AssignNewValue_1?
re_lu_1517/ReluRelu-batch_normalization_1625/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1517/Reluw
flatten_439/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_439/Const?
flatten_439/ReshapeReshapere_lu_1517/Relu:activations:0flatten_439/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_439/Reshape?
dense_878/MatMul/ReadVariableOpReadVariableOp(dense_878_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02!
dense_878/MatMul/ReadVariableOp?
dense_878/MatMulMatMulflatten_439/Reshape:output:0'dense_878/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_878/MatMul?
 dense_878/BiasAdd/ReadVariableOpReadVariableOp)dense_878_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_878/BiasAdd/ReadVariableOp?
dense_878/BiasAddBiasAdddense_878/MatMul:product:0(dense_878/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_878/BiasAddw
dense_878/ReluReludense_878/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_878/Relu?
dense_879/MatMul/ReadVariableOpReadVariableOp(dense_879_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_879/MatMul/ReadVariableOp?
dense_879/MatMulMatMuldense_878/Relu:activations:0'dense_879/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_879/MatMul?
 dense_879/BiasAdd/ReadVariableOpReadVariableOp)dense_879_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_879/BiasAdd/ReadVariableOp?
dense_879/BiasAddBiasAdddense_879/MatMul:product:0(dense_879/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_879/BiasAdd
dense_879/SoftmaxSoftmaxdense_879/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_879/Softmax?
IdentityIdentitydense_879/Softmax:softmax:0&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1(^batch_normalization_1592/AssignNewValue*^batch_normalization_1592/AssignNewValue_19^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1592/ReadVariableOp*^batch_normalization_1592/ReadVariableOp_1(^batch_normalization_1598/AssignNewValue*^batch_normalization_1598/AssignNewValue_19^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1598/ReadVariableOp*^batch_normalization_1598/ReadVariableOp_1(^batch_normalization_1608/AssignNewValue*^batch_normalization_1608/AssignNewValue_19^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1608/ReadVariableOp*^batch_normalization_1608/ReadVariableOp_1(^batch_normalization_1612/AssignNewValue*^batch_normalization_1612/AssignNewValue_19^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1612/ReadVariableOp*^batch_normalization_1612/ReadVariableOp_1(^batch_normalization_1625/AssignNewValue*^batch_normalization_1625/AssignNewValue_19^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1625/ReadVariableOp*^batch_normalization_1625/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp#^conv2d_1592/BiasAdd/ReadVariableOp"^conv2d_1592/Conv2D/ReadVariableOp#^conv2d_1598/BiasAdd/ReadVariableOp"^conv2d_1598/Conv2D/ReadVariableOp#^conv2d_1608/BiasAdd/ReadVariableOp"^conv2d_1608/Conv2D/ReadVariableOp#^conv2d_1612/BiasAdd/ReadVariableOp"^conv2d_1612/Conv2D/ReadVariableOp#^conv2d_1625/BiasAdd/ReadVariableOp"^conv2d_1625/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp!^dense_878/BiasAdd/ReadVariableOp ^dense_878/MatMul/ReadVariableOp!^dense_879/BiasAdd/ReadVariableOp ^dense_879/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12R
'batch_normalization_1592/AssignNewValue'batch_normalization_1592/AssignNewValue2V
)batch_normalization_1592/AssignNewValue_1)batch_normalization_1592/AssignNewValue_12t
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1592/ReadVariableOp'batch_normalization_1592/ReadVariableOp2V
)batch_normalization_1592/ReadVariableOp_1)batch_normalization_1592/ReadVariableOp_12R
'batch_normalization_1598/AssignNewValue'batch_normalization_1598/AssignNewValue2V
)batch_normalization_1598/AssignNewValue_1)batch_normalization_1598/AssignNewValue_12t
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1598/ReadVariableOp'batch_normalization_1598/ReadVariableOp2V
)batch_normalization_1598/ReadVariableOp_1)batch_normalization_1598/ReadVariableOp_12R
'batch_normalization_1608/AssignNewValue'batch_normalization_1608/AssignNewValue2V
)batch_normalization_1608/AssignNewValue_1)batch_normalization_1608/AssignNewValue_12t
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1608/ReadVariableOp'batch_normalization_1608/ReadVariableOp2V
)batch_normalization_1608/ReadVariableOp_1)batch_normalization_1608/ReadVariableOp_12R
'batch_normalization_1612/AssignNewValue'batch_normalization_1612/AssignNewValue2V
)batch_normalization_1612/AssignNewValue_1)batch_normalization_1612/AssignNewValue_12t
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1612/ReadVariableOp'batch_normalization_1612/ReadVariableOp2V
)batch_normalization_1612/ReadVariableOp_1)batch_normalization_1612/ReadVariableOp_12R
'batch_normalization_1625/AssignNewValue'batch_normalization_1625/AssignNewValue2V
)batch_normalization_1625/AssignNewValue_1)batch_normalization_1625/AssignNewValue_12t
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1625/ReadVariableOp'batch_normalization_1625/ReadVariableOp2V
)batch_normalization_1625/ReadVariableOp_1)batch_normalization_1625/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2H
"conv2d_1592/BiasAdd/ReadVariableOp"conv2d_1592/BiasAdd/ReadVariableOp2F
!conv2d_1592/Conv2D/ReadVariableOp!conv2d_1592/Conv2D/ReadVariableOp2H
"conv2d_1598/BiasAdd/ReadVariableOp"conv2d_1598/BiasAdd/ReadVariableOp2F
!conv2d_1598/Conv2D/ReadVariableOp!conv2d_1598/Conv2D/ReadVariableOp2H
"conv2d_1608/BiasAdd/ReadVariableOp"conv2d_1608/BiasAdd/ReadVariableOp2F
!conv2d_1608/Conv2D/ReadVariableOp!conv2d_1608/Conv2D/ReadVariableOp2H
"conv2d_1612/BiasAdd/ReadVariableOp"conv2d_1612/BiasAdd/ReadVariableOp2F
!conv2d_1612/Conv2D/ReadVariableOp!conv2d_1612/Conv2D/ReadVariableOp2H
"conv2d_1625/BiasAdd/ReadVariableOp"conv2d_1625/BiasAdd/ReadVariableOp2F
!conv2d_1625/Conv2D/ReadVariableOp!conv2d_1625/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2D
 dense_878/BiasAdd/ReadVariableOp dense_878/BiasAdd/ReadVariableOp2B
dense_878/MatMul/ReadVariableOpdense_878/MatMul/ReadVariableOp2D
 dense_879/BiasAdd/ReadVariableOp dense_879/BiasAdd/ReadVariableOp2B
dense_879/MatMul/ReadVariableOpdense_879/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_5507301

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_5507764

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_5509517

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5506963

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1592_layer_call_fn_5509983

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_55073542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_34_layer_call_and_return_conditional_losses_5507188

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_15_layer_call_fn_5509669

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_55062542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_34_layer_call_fn_5506381

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_55063752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
U
)__inference_add_765_layer_call_fn_5510162
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_765_layer_call_and_return_conditional_losses_55075232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
n
D__inference_add_767_layer_call_and_return_conditional_losses_5507779

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
??
?.
 __inference__traced_save_5511019
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop1
-savev2_conv2d_1592_kernel_read_readvariableop/
+savev2_conv2d_1592_bias_read_readvariableop=
9savev2_batch_normalization_1592_gamma_read_readvariableop<
8savev2_batch_normalization_1592_beta_read_readvariableopC
?savev2_batch_normalization_1592_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1592_moving_variance_read_readvariableop1
-savev2_conv2d_1598_kernel_read_readvariableop/
+savev2_conv2d_1598_bias_read_readvariableop=
9savev2_batch_normalization_1598_gamma_read_readvariableop<
8savev2_batch_normalization_1598_beta_read_readvariableopC
?savev2_batch_normalization_1598_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1598_moving_variance_read_readvariableop1
-savev2_conv2d_1608_kernel_read_readvariableop/
+savev2_conv2d_1608_bias_read_readvariableop=
9savev2_batch_normalization_1608_gamma_read_readvariableop<
8savev2_batch_normalization_1608_beta_read_readvariableopC
?savev2_batch_normalization_1608_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1608_moving_variance_read_readvariableop1
-savev2_conv2d_1612_kernel_read_readvariableop/
+savev2_conv2d_1612_bias_read_readvariableop=
9savev2_batch_normalization_1612_gamma_read_readvariableop<
8savev2_batch_normalization_1612_beta_read_readvariableopC
?savev2_batch_normalization_1612_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1612_moving_variance_read_readvariableop1
-savev2_conv2d_1625_kernel_read_readvariableop/
+savev2_conv2d_1625_bias_read_readvariableop=
9savev2_batch_normalization_1625_gamma_read_readvariableop<
8savev2_batch_normalization_1625_beta_read_readvariableopC
?savev2_batch_normalization_1625_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1625_moving_variance_read_readvariableop/
+savev2_dense_878_kernel_read_readvariableop-
)savev2_dense_878_bias_read_readvariableop/
+savev2_dense_879_kernel_read_readvariableop-
)savev2_dense_879_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_sgd_conv2d_4_kernel_momentum_read_readvariableop9
5savev2_sgd_conv2d_4_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_4_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_4_beta_momentum_read_readvariableop<
8savev2_sgd_conv2d_15_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_15_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_15_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_15_beta_momentum_read_readvariableop<
8savev2_sgd_conv2d_34_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_34_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_34_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_34_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1592_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1592_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1592_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1592_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1598_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1598_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1598_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1598_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1608_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1608_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1608_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1608_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1612_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1612_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1612_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1612_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1625_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1625_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1625_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1625_beta_momentum_read_readvariableop<
8savev2_sgd_dense_878_kernel_momentum_read_readvariableop:
6savev2_sgd_dense_878_bias_momentum_read_readvariableop<
8savev2_sgd_dense_879_kernel_momentum_read_readvariableop:
6savev2_sgd_dense_879_bias_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?5
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?4
value?4B?4aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop-savev2_conv2d_1592_kernel_read_readvariableop+savev2_conv2d_1592_bias_read_readvariableop9savev2_batch_normalization_1592_gamma_read_readvariableop8savev2_batch_normalization_1592_beta_read_readvariableop?savev2_batch_normalization_1592_moving_mean_read_readvariableopCsavev2_batch_normalization_1592_moving_variance_read_readvariableop-savev2_conv2d_1598_kernel_read_readvariableop+savev2_conv2d_1598_bias_read_readvariableop9savev2_batch_normalization_1598_gamma_read_readvariableop8savev2_batch_normalization_1598_beta_read_readvariableop?savev2_batch_normalization_1598_moving_mean_read_readvariableopCsavev2_batch_normalization_1598_moving_variance_read_readvariableop-savev2_conv2d_1608_kernel_read_readvariableop+savev2_conv2d_1608_bias_read_readvariableop9savev2_batch_normalization_1608_gamma_read_readvariableop8savev2_batch_normalization_1608_beta_read_readvariableop?savev2_batch_normalization_1608_moving_mean_read_readvariableopCsavev2_batch_normalization_1608_moving_variance_read_readvariableop-savev2_conv2d_1612_kernel_read_readvariableop+savev2_conv2d_1612_bias_read_readvariableop9savev2_batch_normalization_1612_gamma_read_readvariableop8savev2_batch_normalization_1612_beta_read_readvariableop?savev2_batch_normalization_1612_moving_mean_read_readvariableopCsavev2_batch_normalization_1612_moving_variance_read_readvariableop-savev2_conv2d_1625_kernel_read_readvariableop+savev2_conv2d_1625_bias_read_readvariableop9savev2_batch_normalization_1625_gamma_read_readvariableop8savev2_batch_normalization_1625_beta_read_readvariableop?savev2_batch_normalization_1625_moving_mean_read_readvariableopCsavev2_batch_normalization_1625_moving_variance_read_readvariableop+savev2_dense_878_kernel_read_readvariableop)savev2_dense_878_bias_read_readvariableop+savev2_dense_879_kernel_read_readvariableop)savev2_dense_879_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_sgd_conv2d_4_kernel_momentum_read_readvariableop5savev2_sgd_conv2d_4_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_4_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_4_beta_momentum_read_readvariableop8savev2_sgd_conv2d_15_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_15_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_15_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_15_beta_momentum_read_readvariableop8savev2_sgd_conv2d_34_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_34_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_34_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_34_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1592_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1592_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1592_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1592_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1598_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1598_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1598_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1598_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1608_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1608_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1608_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1608_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1612_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1612_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1612_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1612_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1625_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1625_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1625_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1625_beta_momentum_read_readvariableop8savev2_sgd_dense_878_kernel_momentum_read_readvariableop6savev2_sgd_dense_878_bias_momentum_read_readvariableop8savev2_sgd_dense_879_kernel_momentum_read_readvariableop6savev2_sgd_dense_879_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *o
dtypese
c2a	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::::: : : : : : :  : : : : : :  : : : : : :  : : : : : :  : : : : : :???:?:	?
:
: : : : : : : : ::::::::::::: : : : :  : : : :  : : : :  : : : :  : : : :???:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:  : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :,+(
&
_output_shapes
:  : ,

_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: :'1#
!
_output_shapes
:???:!2

_output_shapes	
:?:%3!

_output_shapes
:	?
: 4

_output_shapes
:
:5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :,=(
&
_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
::,A(
&
_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
::,E(
&
_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
::,I(
&
_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: :,M(
&
_output_shapes
:  : N

_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: :,Q(
&
_output_shapes
:  : R

_output_shapes
: : S

_output_shapes
: : T

_output_shapes
: :,U(
&
_output_shapes
:  : V

_output_shapes
: : W

_output_shapes
: : X

_output_shapes
: :,Y(
&
_output_shapes
:  : Z

_output_shapes
: : [

_output_shapes
: : \

_output_shapes
: :']#
!
_output_shapes
:???:!^

_output_shapes	
:?:%_!

_output_shapes
:	?
: `

_output_shapes
:
:a

_output_shapes
: 
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510621

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_34_layer_call_fn_5509762

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_55072412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
F__inference_model_438_layer_call_and_return_conditional_losses_5508113
input_1
conv2d_4_5507972
conv2d_4_5507974!
batch_normalization_4_5507977!
batch_normalization_4_5507979!
batch_normalization_4_5507981!
batch_normalization_4_5507983
conv2d_15_5507988
conv2d_15_5507990"
batch_normalization_15_5507993"
batch_normalization_15_5507995"
batch_normalization_15_5507997"
batch_normalization_15_5507999
conv2d_34_5508003
conv2d_34_5508005"
batch_normalization_34_5508008"
batch_normalization_34_5508010"
batch_normalization_34_5508012"
batch_normalization_34_5508014
conv2d_1592_5508019
conv2d_1592_5508021$
 batch_normalization_1592_5508024$
 batch_normalization_1592_5508026$
 batch_normalization_1592_5508028$
 batch_normalization_1592_5508030
conv2d_1598_5508035
conv2d_1598_5508037$
 batch_normalization_1598_5508040$
 batch_normalization_1598_5508042$
 batch_normalization_1598_5508044$
 batch_normalization_1598_5508046
conv2d_1608_5508052
conv2d_1608_5508054$
 batch_normalization_1608_5508057$
 batch_normalization_1608_5508059$
 batch_normalization_1608_5508061$
 batch_normalization_1608_5508063
conv2d_1612_5508069
conv2d_1612_5508071$
 batch_normalization_1612_5508074$
 batch_normalization_1612_5508076$
 batch_normalization_1612_5508078$
 batch_normalization_1612_5508080
conv2d_1625_5508086
conv2d_1625_5508088$
 batch_normalization_1625_5508091$
 batch_normalization_1625_5508093$
 batch_normalization_1625_5508095$
 batch_normalization_1625_5508097
dense_878_5508102
dense_878_5508104
dense_879_5508107
dense_879_5508109
identity??.batch_normalization_15/StatefulPartitionedCall?0batch_normalization_1592/StatefulPartitionedCall?0batch_normalization_1598/StatefulPartitionedCall?0batch_normalization_1608/StatefulPartitionedCall?0batch_normalization_1612/StatefulPartitionedCall?0batch_normalization_1625/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#conv2d_1592/StatefulPartitionedCall?#conv2d_1598/StatefulPartitionedCall?#conv2d_1608/StatefulPartitionedCall?#conv2d_1612/StatefulPartitionedCall?#conv2d_1625/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?!dense_878/StatefulPartitionedCall?!dense_879/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_4_5507972conv2d_4_5507974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_55069632"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_5507977batch_normalization_4_5507979batch_normalization_4_5507981batch_normalization_4_5507983*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55070162/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_55070572
re_lu_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_55061552!
max_pooling2d_4/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_15_5507988conv2d_15_5507990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_55070762#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_5507993batch_normalization_15_5507995batch_normalization_15_5507997batch_normalization_15_5507999*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_550712920
.batch_normalization_15/StatefulPartitionedCall?
re_lu_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_15_layer_call_and_return_conditional_losses_55071702
re_lu_15/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_15/PartitionedCall:output:0conv2d_34_5508003conv2d_34_5508005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_55071882#
!conv2d_34/StatefulPartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_34_5508008batch_normalization_34_5508010batch_normalization_34_5508012batch_normalization_34_5508014*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_550724120
.batch_normalization_34/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_34_layer_call_and_return_conditional_losses_55072822
re_lu_34/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_55063752"
 max_pooling2d_34/PartitionedCall?
#conv2d_1592/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_1592_5508019conv2d_1592_5508021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_55073012%
#conv2d_1592/StatefulPartitionedCall?
0batch_normalization_1592/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1592/StatefulPartitionedCall:output:0 batch_normalization_1592_5508024 batch_normalization_1592_5508026 batch_normalization_1592_5508028 batch_normalization_1592_5508030*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_550735422
0batch_normalization_1592/StatefulPartitionedCall?
re_lu_1484/PartitionedCallPartitionedCall9batch_normalization_1592/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_55073952
re_lu_1484/PartitionedCall?
"max_pooling2d_1484/PartitionedCallPartitionedCall#re_lu_1484/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_55064912$
"max_pooling2d_1484/PartitionedCall?
#conv2d_1598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0conv2d_1598_5508035conv2d_1598_5508037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_55074142%
#conv2d_1598/StatefulPartitionedCall?
0batch_normalization_1598/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1598/StatefulPartitionedCall:output:0 batch_normalization_1598_5508040 batch_normalization_1598_5508042 batch_normalization_1598_5508044 batch_normalization_1598_5508046*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_550746722
0batch_normalization_1598/StatefulPartitionedCall?
re_lu_1490/PartitionedCallPartitionedCall9batch_normalization_1598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_55075082
re_lu_1490/PartitionedCall?
"max_pooling2d_1490/PartitionedCallPartitionedCall#re_lu_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_55066072$
"max_pooling2d_1490/PartitionedCall?
add_765/PartitionedCallPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_765_layer_call_and_return_conditional_losses_55075232
add_765/PartitionedCall?
#conv2d_1608/StatefulPartitionedCallStatefulPartitionedCall add_765/PartitionedCall:output:0conv2d_1608_5508052conv2d_1608_5508054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_55075422%
#conv2d_1608/StatefulPartitionedCall?
0batch_normalization_1608/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1608/StatefulPartitionedCall:output:0 batch_normalization_1608_5508057 batch_normalization_1608_5508059 batch_normalization_1608_5508061 batch_normalization_1608_5508063*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_550759522
0batch_normalization_1608/StatefulPartitionedCall?
re_lu_1500/PartitionedCallPartitionedCall9batch_normalization_1608/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_55076362
re_lu_1500/PartitionedCall?
"max_pooling2d_1500/PartitionedCallPartitionedCall#re_lu_1500/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_55067232$
"max_pooling2d_1500/PartitionedCall?
add_766/PartitionedCallPartitionedCall+max_pooling2d_1500/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_766_layer_call_and_return_conditional_losses_55076512
add_766/PartitionedCall?
#conv2d_1612/StatefulPartitionedCallStatefulPartitionedCall add_766/PartitionedCall:output:0conv2d_1612_5508069conv2d_1612_5508071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_55076702%
#conv2d_1612/StatefulPartitionedCall?
0batch_normalization_1612/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1612/StatefulPartitionedCall:output:0 batch_normalization_1612_5508074 batch_normalization_1612_5508076 batch_normalization_1612_5508078 batch_normalization_1612_5508080*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_550772322
0batch_normalization_1612/StatefulPartitionedCall?
re_lu_1504/PartitionedCallPartitionedCall9batch_normalization_1612/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_55077642
re_lu_1504/PartitionedCall?
"max_pooling2d_1504/PartitionedCallPartitionedCall#re_lu_1504/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_55068392$
"max_pooling2d_1504/PartitionedCall?
add_767/PartitionedCallPartitionedCall+max_pooling2d_1504/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_767_layer_call_and_return_conditional_losses_55077792
add_767/PartitionedCall?
#conv2d_1625/StatefulPartitionedCallStatefulPartitionedCall add_767/PartitionedCall:output:0conv2d_1625_5508086conv2d_1625_5508088*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_55077982%
#conv2d_1625/StatefulPartitionedCall?
0batch_normalization_1625/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1625/StatefulPartitionedCall:output:0 batch_normalization_1625_5508091 batch_normalization_1625_5508093 batch_normalization_1625_5508095 batch_normalization_1625_5508097*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_550785122
0batch_normalization_1625/StatefulPartitionedCall?
re_lu_1517/PartitionedCallPartitionedCall9batch_normalization_1625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_55078922
re_lu_1517/PartitionedCall?
flatten_439/PartitionedCallPartitionedCall#re_lu_1517/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_439_layer_call_and_return_conditional_losses_55079062
flatten_439/PartitionedCall?
!dense_878/StatefulPartitionedCallStatefulPartitionedCall$flatten_439/PartitionedCall:output:0dense_878_5508102dense_878_5508104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_55079252#
!dense_878/StatefulPartitionedCall?
!dense_879/StatefulPartitionedCallStatefulPartitionedCall*dense_878/StatefulPartitionedCall:output:0dense_879_5508107dense_879_5508109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_879_layer_call_and_return_conditional_losses_55079522#
!dense_879/StatefulPartitionedCall?
IdentityIdentity*dense_879/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall1^batch_normalization_1592/StatefulPartitionedCall1^batch_normalization_1598/StatefulPartitionedCall1^batch_normalization_1608/StatefulPartitionedCall1^batch_normalization_1612/StatefulPartitionedCall1^batch_normalization_1625/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^conv2d_1592/StatefulPartitionedCall$^conv2d_1598/StatefulPartitionedCall$^conv2d_1608/StatefulPartitionedCall$^conv2d_1612/StatefulPartitionedCall$^conv2d_1625/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall"^dense_879/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2d
0batch_normalization_1592/StatefulPartitionedCall0batch_normalization_1592/StatefulPartitionedCall2d
0batch_normalization_1598/StatefulPartitionedCall0batch_normalization_1598/StatefulPartitionedCall2d
0batch_normalization_1608/StatefulPartitionedCall0batch_normalization_1608/StatefulPartitionedCall2d
0batch_normalization_1612/StatefulPartitionedCall0batch_normalization_1612/StatefulPartitionedCall2d
0batch_normalization_1625/StatefulPartitionedCall0batch_normalization_1625/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#conv2d_1592/StatefulPartitionedCall#conv2d_1592/StatefulPartitionedCall2J
#conv2d_1598/StatefulPartitionedCall#conv2d_1598/StatefulPartitionedCall2J
#conv2d_1608/StatefulPartitionedCall#conv2d_1608/StatefulPartitionedCall2J
#conv2d_1612/StatefulPartitionedCall#conv2d_1612/StatefulPartitionedCall2J
#conv2d_1625/StatefulPartitionedCall#conv2d_1625/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall2F
!dense_879/StatefulPartitionedCall!dense_879/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5507467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_5510652

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1625_layer_call_fn_5510570

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_55078332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_4_layer_call_fn_5509512

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55061382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_34_layer_call_and_return_conditional_losses_5509831

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510603

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

*__inference_conv2d_4_layer_call_fn_5509384

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_55069632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_5507670

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
F__inference_dense_878_layer_call_and_return_conditional_losses_5510679

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_5510145

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_5507798

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510557

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1592_layer_call_fn_5509855

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_55073012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1612_layer_call_fn_5510414

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_55077232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_5510172

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1612_layer_call_fn_5510478

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_55068222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_re_lu_15_layer_call_and_return_conditional_losses_5509674

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?9
#__inference__traced_restore_5511317
file_prefix$
 assignvariableop_conv2d_4_kernel$
 assignvariableop_1_conv2d_4_bias2
.assignvariableop_2_batch_normalization_4_gamma1
-assignvariableop_3_batch_normalization_4_beta8
4assignvariableop_4_batch_normalization_4_moving_mean<
8assignvariableop_5_batch_normalization_4_moving_variance'
#assignvariableop_6_conv2d_15_kernel%
!assignvariableop_7_conv2d_15_bias3
/assignvariableop_8_batch_normalization_15_gamma2
.assignvariableop_9_batch_normalization_15_beta:
6assignvariableop_10_batch_normalization_15_moving_mean>
:assignvariableop_11_batch_normalization_15_moving_variance(
$assignvariableop_12_conv2d_34_kernel&
"assignvariableop_13_conv2d_34_bias4
0assignvariableop_14_batch_normalization_34_gamma3
/assignvariableop_15_batch_normalization_34_beta:
6assignvariableop_16_batch_normalization_34_moving_mean>
:assignvariableop_17_batch_normalization_34_moving_variance*
&assignvariableop_18_conv2d_1592_kernel(
$assignvariableop_19_conv2d_1592_bias6
2assignvariableop_20_batch_normalization_1592_gamma5
1assignvariableop_21_batch_normalization_1592_beta<
8assignvariableop_22_batch_normalization_1592_moving_mean@
<assignvariableop_23_batch_normalization_1592_moving_variance*
&assignvariableop_24_conv2d_1598_kernel(
$assignvariableop_25_conv2d_1598_bias6
2assignvariableop_26_batch_normalization_1598_gamma5
1assignvariableop_27_batch_normalization_1598_beta<
8assignvariableop_28_batch_normalization_1598_moving_mean@
<assignvariableop_29_batch_normalization_1598_moving_variance*
&assignvariableop_30_conv2d_1608_kernel(
$assignvariableop_31_conv2d_1608_bias6
2assignvariableop_32_batch_normalization_1608_gamma5
1assignvariableop_33_batch_normalization_1608_beta<
8assignvariableop_34_batch_normalization_1608_moving_mean@
<assignvariableop_35_batch_normalization_1608_moving_variance*
&assignvariableop_36_conv2d_1612_kernel(
$assignvariableop_37_conv2d_1612_bias6
2assignvariableop_38_batch_normalization_1612_gamma5
1assignvariableop_39_batch_normalization_1612_beta<
8assignvariableop_40_batch_normalization_1612_moving_mean@
<assignvariableop_41_batch_normalization_1612_moving_variance*
&assignvariableop_42_conv2d_1625_kernel(
$assignvariableop_43_conv2d_1625_bias6
2assignvariableop_44_batch_normalization_1625_gamma5
1assignvariableop_45_batch_normalization_1625_beta<
8assignvariableop_46_batch_normalization_1625_moving_mean@
<assignvariableop_47_batch_normalization_1625_moving_variance(
$assignvariableop_48_dense_878_kernel&
"assignvariableop_49_dense_878_bias(
$assignvariableop_50_dense_879_kernel&
"assignvariableop_51_dense_879_bias 
assignvariableop_52_sgd_iter!
assignvariableop_53_sgd_decay)
%assignvariableop_54_sgd_learning_rate$
 assignvariableop_55_sgd_momentum
assignvariableop_56_total
assignvariableop_57_count
assignvariableop_58_total_1
assignvariableop_59_count_14
0assignvariableop_60_sgd_conv2d_4_kernel_momentum2
.assignvariableop_61_sgd_conv2d_4_bias_momentum@
<assignvariableop_62_sgd_batch_normalization_4_gamma_momentum?
;assignvariableop_63_sgd_batch_normalization_4_beta_momentum5
1assignvariableop_64_sgd_conv2d_15_kernel_momentum3
/assignvariableop_65_sgd_conv2d_15_bias_momentumA
=assignvariableop_66_sgd_batch_normalization_15_gamma_momentum@
<assignvariableop_67_sgd_batch_normalization_15_beta_momentum5
1assignvariableop_68_sgd_conv2d_34_kernel_momentum3
/assignvariableop_69_sgd_conv2d_34_bias_momentumA
=assignvariableop_70_sgd_batch_normalization_34_gamma_momentum@
<assignvariableop_71_sgd_batch_normalization_34_beta_momentum7
3assignvariableop_72_sgd_conv2d_1592_kernel_momentum5
1assignvariableop_73_sgd_conv2d_1592_bias_momentumC
?assignvariableop_74_sgd_batch_normalization_1592_gamma_momentumB
>assignvariableop_75_sgd_batch_normalization_1592_beta_momentum7
3assignvariableop_76_sgd_conv2d_1598_kernel_momentum5
1assignvariableop_77_sgd_conv2d_1598_bias_momentumC
?assignvariableop_78_sgd_batch_normalization_1598_gamma_momentumB
>assignvariableop_79_sgd_batch_normalization_1598_beta_momentum7
3assignvariableop_80_sgd_conv2d_1608_kernel_momentum5
1assignvariableop_81_sgd_conv2d_1608_bias_momentumC
?assignvariableop_82_sgd_batch_normalization_1608_gamma_momentumB
>assignvariableop_83_sgd_batch_normalization_1608_beta_momentum7
3assignvariableop_84_sgd_conv2d_1612_kernel_momentum5
1assignvariableop_85_sgd_conv2d_1612_bias_momentumC
?assignvariableop_86_sgd_batch_normalization_1612_gamma_momentumB
>assignvariableop_87_sgd_batch_normalization_1612_beta_momentum7
3assignvariableop_88_sgd_conv2d_1625_kernel_momentum5
1assignvariableop_89_sgd_conv2d_1625_bias_momentumC
?assignvariableop_90_sgd_batch_normalization_1625_gamma_momentumB
>assignvariableop_91_sgd_batch_normalization_1625_beta_momentum5
1assignvariableop_92_sgd_dense_878_kernel_momentum3
/assignvariableop_93_sgd_dense_878_bias_momentum5
1assignvariableop_94_sgd_dense_879_kernel_momentum3
/assignvariableop_95_sgd_dense_879_bias_momentum
identity_97??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?4
value?4B?4aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_4_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_4_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_4_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_4_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_15_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_15_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_15_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_15_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_15_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_34_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_34_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_34_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_34_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_34_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_34_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_conv2d_1592_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_1592_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp2assignvariableop_20_batch_normalization_1592_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_batch_normalization_1592_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp8assignvariableop_22_batch_normalization_1592_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp<assignvariableop_23_batch_normalization_1592_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_conv2d_1598_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_1598_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_batch_normalization_1598_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_batch_normalization_1598_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp8assignvariableop_28_batch_normalization_1598_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp<assignvariableop_29_batch_normalization_1598_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_conv2d_1608_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv2d_1608_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_batch_normalization_1608_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp1assignvariableop_33_batch_normalization_1608_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_batch_normalization_1608_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp<assignvariableop_35_batch_normalization_1608_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_conv2d_1612_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp$assignvariableop_37_conv2d_1612_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_batch_normalization_1612_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp1assignvariableop_39_batch_normalization_1612_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp8assignvariableop_40_batch_normalization_1612_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp<assignvariableop_41_batch_normalization_1612_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_conv2d_1625_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp$assignvariableop_43_conv2d_1625_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_batch_normalization_1625_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp1assignvariableop_45_batch_normalization_1625_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp8assignvariableop_46_batch_normalization_1625_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp<assignvariableop_47_batch_normalization_1625_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_dense_878_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_dense_878_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp$assignvariableop_50_dense_879_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp"assignvariableop_51_dense_879_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_sgd_iterIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_sgd_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp%assignvariableop_54_sgd_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp assignvariableop_55_sgd_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_totalIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_countIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp0assignvariableop_60_sgd_conv2d_4_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp.assignvariableop_61_sgd_conv2d_4_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp<assignvariableop_62_sgd_batch_normalization_4_gamma_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp;assignvariableop_63_sgd_batch_normalization_4_beta_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp1assignvariableop_64_sgd_conv2d_15_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp/assignvariableop_65_sgd_conv2d_15_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp=assignvariableop_66_sgd_batch_normalization_15_gamma_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp<assignvariableop_67_sgd_batch_normalization_15_beta_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp1assignvariableop_68_sgd_conv2d_34_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp/assignvariableop_69_sgd_conv2d_34_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp=assignvariableop_70_sgd_batch_normalization_34_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp<assignvariableop_71_sgd_batch_normalization_34_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp3assignvariableop_72_sgd_conv2d_1592_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp1assignvariableop_73_sgd_conv2d_1592_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp?assignvariableop_74_sgd_batch_normalization_1592_gamma_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp>assignvariableop_75_sgd_batch_normalization_1592_beta_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp3assignvariableop_76_sgd_conv2d_1598_kernel_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp1assignvariableop_77_sgd_conv2d_1598_bias_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp?assignvariableop_78_sgd_batch_normalization_1598_gamma_momentumIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp>assignvariableop_79_sgd_batch_normalization_1598_beta_momentumIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_sgd_conv2d_1608_kernel_momentumIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp1assignvariableop_81_sgd_conv2d_1608_bias_momentumIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp?assignvariableop_82_sgd_batch_normalization_1608_gamma_momentumIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp>assignvariableop_83_sgd_batch_normalization_1608_beta_momentumIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp3assignvariableop_84_sgd_conv2d_1612_kernel_momentumIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp1assignvariableop_85_sgd_conv2d_1612_bias_momentumIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp?assignvariableop_86_sgd_batch_normalization_1612_gamma_momentumIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp>assignvariableop_87_sgd_batch_normalization_1612_beta_momentumIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp3assignvariableop_88_sgd_conv2d_1625_kernel_momentumIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp1assignvariableop_89_sgd_conv2d_1625_bias_momentumIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp?assignvariableop_90_sgd_batch_normalization_1625_gamma_momentumIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp>assignvariableop_91_sgd_batch_normalization_1625_beta_momentumIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp1assignvariableop_92_sgd_dense_878_kernel_momentumIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp/assignvariableop_93_sgd_dense_878_bias_momentumIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp1assignvariableop_94_sgd_dense_879_kernel_momentumIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp/assignvariableop_95_sgd_dense_879_bias_momentumIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_959
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_96Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_96?
Identity_97IdentityIdentity_96:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95*
T0*
_output_shapes
: 2
Identity_97"#
identity_97Identity_97:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_95:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
k
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_5506723

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_34_layer_call_fn_5509749

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_55072232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509939

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
P
4__inference_max_pooling2d_1504_layer_call_fn_5506845

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_55068392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510032

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5506559

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
H
,__inference_re_lu_1490_layer_call_fn_5510150

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_55075082
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
I
-__inference_flatten_439_layer_call_fn_5510668

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_439_layer_call_and_return_conditional_losses_55079062
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5506327

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_4_layer_call_fn_5509435

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55069982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5507111

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509486

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1608_layer_call_fn_5510309

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_55075952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5506443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_5510003

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
F
*__inference_re_lu_15_layer_call_fn_5509679

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_15_layer_call_and_return_conditional_losses_55071702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5506590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1598_layer_call_fn_5510063

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_55074492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5507705

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509718

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
+__inference_model_438_layer_call_fn_5509256

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-.1234*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_438_layer_call_and_return_conditional_losses_55082602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509404

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_5507892

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_5509846

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5506675

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5507336

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510434

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1598_layer_call_fn_5510076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_55074672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5509532

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
d
H__inference_flatten_439_layer_call_and_return_conditional_losses_5507906

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_15_layer_call_fn_5509605

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_55071292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5506358

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509800

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_5507636

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5506791

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5506223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5506998

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
D__inference_add_765_layer_call_and_return_conditional_losses_5507523

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1592_layer_call_fn_5509970

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_55073362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5509375

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
D__inference_add_766_layer_call_and_return_conditional_losses_5507651

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
F__inference_dense_879_layer_call_and_return_conditional_losses_5510699

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_5509988

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
p
D__inference_add_767_layer_call_and_return_conditional_losses_5510494
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
F
*__inference_re_lu_34_layer_call_fn_5509836

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_34_layer_call_and_return_conditional_losses_55072822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5507016

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?*
F__inference_model_438_layer_call_and_return_conditional_losses_5509147

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource2
.batch_normalization_34_readvariableop_resource4
0batch_normalization_34_readvariableop_1_resourceC
?batch_normalization_34_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1592_conv2d_readvariableop_resource/
+conv2d_1592_biasadd_readvariableop_resource4
0batch_normalization_1592_readvariableop_resource6
2batch_normalization_1592_readvariableop_1_resourceE
Abatch_normalization_1592_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1598_conv2d_readvariableop_resource/
+conv2d_1598_biasadd_readvariableop_resource4
0batch_normalization_1598_readvariableop_resource6
2batch_normalization_1598_readvariableop_1_resourceE
Abatch_normalization_1598_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1608_conv2d_readvariableop_resource/
+conv2d_1608_biasadd_readvariableop_resource4
0batch_normalization_1608_readvariableop_resource6
2batch_normalization_1608_readvariableop_1_resourceE
Abatch_normalization_1608_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1612_conv2d_readvariableop_resource/
+conv2d_1612_biasadd_readvariableop_resource4
0batch_normalization_1612_readvariableop_resource6
2batch_normalization_1612_readvariableop_1_resourceE
Abatch_normalization_1612_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1625_conv2d_readvariableop_resource/
+conv2d_1625_biasadd_readvariableop_resource4
0batch_normalization_1625_readvariableop_resource6
2batch_normalization_1625_readvariableop_1_resourceE
Abatch_normalization_1625_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource,
(dense_878_matmul_readvariableop_resource-
)dense_878_biasadd_readvariableop_resource,
(dense_879_matmul_readvariableop_resource-
)dense_879_biasadd_readvariableop_resource
identity??6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1592/ReadVariableOp?)batch_normalization_1592/ReadVariableOp_1?8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1598/ReadVariableOp?)batch_normalization_1598/ReadVariableOp_1?8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1608/ReadVariableOp?)batch_normalization_1608/ReadVariableOp_1?8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1612/ReadVariableOp?)batch_normalization_1612/ReadVariableOp_1?8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1625/ReadVariableOp?)batch_normalization_1625/ReadVariableOp_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?"conv2d_1592/BiasAdd/ReadVariableOp?!conv2d_1592/Conv2D/ReadVariableOp?"conv2d_1598/BiasAdd/ReadVariableOp?!conv2d_1598/Conv2D/ReadVariableOp?"conv2d_1608/BiasAdd/ReadVariableOp?!conv2d_1608/Conv2D/ReadVariableOp?"conv2d_1612/BiasAdd/ReadVariableOp?!conv2d_1612/Conv2D/ReadVariableOp?"conv2d_1625/BiasAdd/ReadVariableOp?!conv2d_1625/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp? dense_878/BiasAdd/ReadVariableOp?dense_878/MatMul/ReadVariableOp? dense_879/BiasAdd/ReadVariableOp?dense_879/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_4/Relu?
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_15/BiasAdd?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
re_lu_15/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_15/Relu?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2Dre_lu_15/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_34/BiasAdd?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_34/ReadVariableOp?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_34/ReadVariableOp_1?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_34/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_34/FusedBatchNormV3?
re_lu_34/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_34/Relu?
max_pooling2d_34/MaxPoolMaxPoolre_lu_34/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_34/MaxPool?
!conv2d_1592/Conv2D/ReadVariableOpReadVariableOp*conv2d_1592_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_1592/Conv2D/ReadVariableOp?
conv2d_1592/Conv2DConv2D!max_pooling2d_34/MaxPool:output:0)conv2d_1592/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1592/Conv2D?
"conv2d_1592/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1592_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1592/BiasAdd/ReadVariableOp?
conv2d_1592/BiasAddBiasAddconv2d_1592/Conv2D:output:0*conv2d_1592/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1592/BiasAdd?
'batch_normalization_1592/ReadVariableOpReadVariableOp0batch_normalization_1592_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1592/ReadVariableOp?
)batch_normalization_1592/ReadVariableOp_1ReadVariableOp2batch_normalization_1592_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1592/ReadVariableOp_1?
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1592_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1592_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1592/FusedBatchNormV3FusedBatchNormV3conv2d_1592/BiasAdd:output:0/batch_normalization_1592/ReadVariableOp:value:01batch_normalization_1592/ReadVariableOp_1:value:0@batch_normalization_1592/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1592/FusedBatchNormV3?
re_lu_1484/ReluRelu-batch_normalization_1592/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1484/Relu?
max_pooling2d_1484/MaxPoolMaxPoolre_lu_1484/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1484/MaxPool?
!conv2d_1598/Conv2D/ReadVariableOpReadVariableOp*conv2d_1598_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1598/Conv2D/ReadVariableOp?
conv2d_1598/Conv2DConv2D#max_pooling2d_1484/MaxPool:output:0)conv2d_1598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1598/Conv2D?
"conv2d_1598/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1598_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1598/BiasAdd/ReadVariableOp?
conv2d_1598/BiasAddBiasAddconv2d_1598/Conv2D:output:0*conv2d_1598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1598/BiasAdd?
'batch_normalization_1598/ReadVariableOpReadVariableOp0batch_normalization_1598_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1598/ReadVariableOp?
)batch_normalization_1598/ReadVariableOp_1ReadVariableOp2batch_normalization_1598_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1598/ReadVariableOp_1?
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1598_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1598_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1598/FusedBatchNormV3FusedBatchNormV3conv2d_1598/BiasAdd:output:0/batch_normalization_1598/ReadVariableOp:value:01batch_normalization_1598/ReadVariableOp_1:value:0@batch_normalization_1598/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1598/FusedBatchNormV3?
re_lu_1490/ReluRelu-batch_normalization_1598/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1490/Relu?
max_pooling2d_1490/MaxPoolMaxPoolre_lu_1490/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1490/MaxPool?
add_765/addAddV2#max_pooling2d_1484/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_765/add?
!conv2d_1608/Conv2D/ReadVariableOpReadVariableOp*conv2d_1608_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1608/Conv2D/ReadVariableOp?
conv2d_1608/Conv2DConv2Dadd_765/add:z:0)conv2d_1608/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1608/Conv2D?
"conv2d_1608/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1608_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1608/BiasAdd/ReadVariableOp?
conv2d_1608/BiasAddBiasAddconv2d_1608/Conv2D:output:0*conv2d_1608/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1608/BiasAdd?
'batch_normalization_1608/ReadVariableOpReadVariableOp0batch_normalization_1608_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1608/ReadVariableOp?
)batch_normalization_1608/ReadVariableOp_1ReadVariableOp2batch_normalization_1608_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1608/ReadVariableOp_1?
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1608_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1608_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1608/FusedBatchNormV3FusedBatchNormV3conv2d_1608/BiasAdd:output:0/batch_normalization_1608/ReadVariableOp:value:01batch_normalization_1608/ReadVariableOp_1:value:0@batch_normalization_1608/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1608/FusedBatchNormV3?
re_lu_1500/ReluRelu-batch_normalization_1608/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1500/Relu?
max_pooling2d_1500/MaxPoolMaxPoolre_lu_1500/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1500/MaxPool?
add_766/addAddV2#max_pooling2d_1500/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_766/add?
!conv2d_1612/Conv2D/ReadVariableOpReadVariableOp*conv2d_1612_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1612/Conv2D/ReadVariableOp?
conv2d_1612/Conv2DConv2Dadd_766/add:z:0)conv2d_1612/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1612/Conv2D?
"conv2d_1612/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1612_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1612/BiasAdd/ReadVariableOp?
conv2d_1612/BiasAddBiasAddconv2d_1612/Conv2D:output:0*conv2d_1612/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1612/BiasAdd?
'batch_normalization_1612/ReadVariableOpReadVariableOp0batch_normalization_1612_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1612/ReadVariableOp?
)batch_normalization_1612/ReadVariableOp_1ReadVariableOp2batch_normalization_1612_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1612/ReadVariableOp_1?
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1612_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1612_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1612/FusedBatchNormV3FusedBatchNormV3conv2d_1612/BiasAdd:output:0/batch_normalization_1612/ReadVariableOp:value:01batch_normalization_1612/ReadVariableOp_1:value:0@batch_normalization_1612/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1612/FusedBatchNormV3?
re_lu_1504/ReluRelu-batch_normalization_1612/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1504/Relu?
max_pooling2d_1504/MaxPoolMaxPoolre_lu_1504/Relu:activations:0*/
_output_shapes
:?????????   *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1504/MaxPool?
add_767/addAddV2#max_pooling2d_1504/MaxPool:output:0#max_pooling2d_1490/MaxPool:output:0*
T0*/
_output_shapes
:?????????   2
add_767/add?
!conv2d_1625/Conv2D/ReadVariableOpReadVariableOp*conv2d_1625_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1625/Conv2D/ReadVariableOp?
conv2d_1625/Conv2DConv2Dadd_767/add:z:0)conv2d_1625/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1625/Conv2D?
"conv2d_1625/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1625_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1625/BiasAdd/ReadVariableOp?
conv2d_1625/BiasAddBiasAddconv2d_1625/Conv2D:output:0*conv2d_1625/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1625/BiasAdd?
'batch_normalization_1625/ReadVariableOpReadVariableOp0batch_normalization_1625_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1625/ReadVariableOp?
)batch_normalization_1625/ReadVariableOp_1ReadVariableOp2batch_normalization_1625_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1625/ReadVariableOp_1?
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1625_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1625_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1625/FusedBatchNormV3FusedBatchNormV3conv2d_1625/BiasAdd:output:0/batch_normalization_1625/ReadVariableOp:value:01batch_normalization_1625/ReadVariableOp_1:value:0@batch_normalization_1625/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1625/FusedBatchNormV3?
re_lu_1517/ReluRelu-batch_normalization_1625/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1517/Reluw
flatten_439/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_439/Const?
flatten_439/ReshapeReshapere_lu_1517/Relu:activations:0flatten_439/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_439/Reshape?
dense_878/MatMul/ReadVariableOpReadVariableOp(dense_878_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02!
dense_878/MatMul/ReadVariableOp?
dense_878/MatMulMatMulflatten_439/Reshape:output:0'dense_878/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_878/MatMul?
 dense_878/BiasAdd/ReadVariableOpReadVariableOp)dense_878_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_878/BiasAdd/ReadVariableOp?
dense_878/BiasAddBiasAdddense_878/MatMul:product:0(dense_878/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_878/BiasAddw
dense_878/ReluReludense_878/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_878/Relu?
dense_879/MatMul/ReadVariableOpReadVariableOp(dense_879_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_879/MatMul/ReadVariableOp?
dense_879/MatMulMatMuldense_878/Relu:activations:0'dense_879/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_879/MatMul?
 dense_879/BiasAdd/ReadVariableOpReadVariableOp)dense_879_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_879/BiasAdd/ReadVariableOp?
dense_879/BiasAddBiasAdddense_879/MatMul:product:0(dense_879/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_879/BiasAdd
dense_879/SoftmaxSoftmaxdense_879/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_879/Softmax?
IdentityIdentitydense_879/Softmax:softmax:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_19^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1592/ReadVariableOp*^batch_normalization_1592/ReadVariableOp_19^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1598/ReadVariableOp*^batch_normalization_1598/ReadVariableOp_19^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1608/ReadVariableOp*^batch_normalization_1608/ReadVariableOp_19^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1612/ReadVariableOp*^batch_normalization_1612/ReadVariableOp_19^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1625/ReadVariableOp*^batch_normalization_1625/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp#^conv2d_1592/BiasAdd/ReadVariableOp"^conv2d_1592/Conv2D/ReadVariableOp#^conv2d_1598/BiasAdd/ReadVariableOp"^conv2d_1598/Conv2D/ReadVariableOp#^conv2d_1608/BiasAdd/ReadVariableOp"^conv2d_1608/Conv2D/ReadVariableOp#^conv2d_1612/BiasAdd/ReadVariableOp"^conv2d_1612/Conv2D/ReadVariableOp#^conv2d_1625/BiasAdd/ReadVariableOp"^conv2d_1625/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp!^dense_878/BiasAdd/ReadVariableOp ^dense_878/MatMul/ReadVariableOp!^dense_879/BiasAdd/ReadVariableOp ^dense_879/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12t
8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp8batch_normalization_1592/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1592/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1592/ReadVariableOp'batch_normalization_1592/ReadVariableOp2V
)batch_normalization_1592/ReadVariableOp_1)batch_normalization_1592/ReadVariableOp_12t
8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp8batch_normalization_1598/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1598/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1598/ReadVariableOp'batch_normalization_1598/ReadVariableOp2V
)batch_normalization_1598/ReadVariableOp_1)batch_normalization_1598/ReadVariableOp_12t
8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp8batch_normalization_1608/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1608/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1608/ReadVariableOp'batch_normalization_1608/ReadVariableOp2V
)batch_normalization_1608/ReadVariableOp_1)batch_normalization_1608/ReadVariableOp_12t
8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp8batch_normalization_1612/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1612/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1612/ReadVariableOp'batch_normalization_1612/ReadVariableOp2V
)batch_normalization_1612/ReadVariableOp_1)batch_normalization_1612/ReadVariableOp_12t
8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp8batch_normalization_1625/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1625/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1625/ReadVariableOp'batch_normalization_1625/ReadVariableOp2V
)batch_normalization_1625/ReadVariableOp_1)batch_normalization_1625/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2H
"conv2d_1592/BiasAdd/ReadVariableOp"conv2d_1592/BiasAdd/ReadVariableOp2F
!conv2d_1592/Conv2D/ReadVariableOp!conv2d_1592/Conv2D/ReadVariableOp2H
"conv2d_1598/BiasAdd/ReadVariableOp"conv2d_1598/BiasAdd/ReadVariableOp2F
!conv2d_1598/Conv2D/ReadVariableOp!conv2d_1598/Conv2D/ReadVariableOp2H
"conv2d_1608/BiasAdd/ReadVariableOp"conv2d_1608/BiasAdd/ReadVariableOp2F
!conv2d_1608/Conv2D/ReadVariableOp!conv2d_1608/Conv2D/ReadVariableOp2H
"conv2d_1612/BiasAdd/ReadVariableOp"conv2d_1612/BiasAdd/ReadVariableOp2F
!conv2d_1612/Conv2D/ReadVariableOp!conv2d_1612/Conv2D/ReadVariableOp2H
"conv2d_1625/BiasAdd/ReadVariableOp"conv2d_1625/BiasAdd/ReadVariableOp2F
!conv2d_1625/Conv2D/ReadVariableOp!conv2d_1625/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2D
 dense_878/BiasAdd/ReadVariableOp dense_878/BiasAdd/ReadVariableOp2B
dense_878/MatMul/ReadVariableOpdense_878/MatMul/ReadVariableOp2D
 dense_879/BiasAdd/ReadVariableOp dense_879/BiasAdd/ReadVariableOp2B
dense_879/MatMul/ReadVariableOpdense_879/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5506474

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_5508737
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_55060452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
:__inference_batch_normalization_1592_layer_call_fn_5509906

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_55064432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
H__inference_flatten_439_layer_call_and_return_conditional_losses_5510663

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509875

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1625_layer_call_fn_5510519

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_55077982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509643

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_34_layer_call_fn_5509826

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_55063582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5507223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510539

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_5510341

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_34_layer_call_and_return_conditional_losses_5509689

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
F__inference_model_438_layer_call_and_return_conditional_losses_5508513

inputs
conv2d_4_5508372
conv2d_4_5508374!
batch_normalization_4_5508377!
batch_normalization_4_5508379!
batch_normalization_4_5508381!
batch_normalization_4_5508383
conv2d_15_5508388
conv2d_15_5508390"
batch_normalization_15_5508393"
batch_normalization_15_5508395"
batch_normalization_15_5508397"
batch_normalization_15_5508399
conv2d_34_5508403
conv2d_34_5508405"
batch_normalization_34_5508408"
batch_normalization_34_5508410"
batch_normalization_34_5508412"
batch_normalization_34_5508414
conv2d_1592_5508419
conv2d_1592_5508421$
 batch_normalization_1592_5508424$
 batch_normalization_1592_5508426$
 batch_normalization_1592_5508428$
 batch_normalization_1592_5508430
conv2d_1598_5508435
conv2d_1598_5508437$
 batch_normalization_1598_5508440$
 batch_normalization_1598_5508442$
 batch_normalization_1598_5508444$
 batch_normalization_1598_5508446
conv2d_1608_5508452
conv2d_1608_5508454$
 batch_normalization_1608_5508457$
 batch_normalization_1608_5508459$
 batch_normalization_1608_5508461$
 batch_normalization_1608_5508463
conv2d_1612_5508469
conv2d_1612_5508471$
 batch_normalization_1612_5508474$
 batch_normalization_1612_5508476$
 batch_normalization_1612_5508478$
 batch_normalization_1612_5508480
conv2d_1625_5508486
conv2d_1625_5508488$
 batch_normalization_1625_5508491$
 batch_normalization_1625_5508493$
 batch_normalization_1625_5508495$
 batch_normalization_1625_5508497
dense_878_5508502
dense_878_5508504
dense_879_5508507
dense_879_5508509
identity??.batch_normalization_15/StatefulPartitionedCall?0batch_normalization_1592/StatefulPartitionedCall?0batch_normalization_1598/StatefulPartitionedCall?0batch_normalization_1608/StatefulPartitionedCall?0batch_normalization_1612/StatefulPartitionedCall?0batch_normalization_1625/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#conv2d_1592/StatefulPartitionedCall?#conv2d_1598/StatefulPartitionedCall?#conv2d_1608/StatefulPartitionedCall?#conv2d_1612/StatefulPartitionedCall?#conv2d_1625/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?!dense_878/StatefulPartitionedCall?!dense_879/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_5508372conv2d_4_5508374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_55069632"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_5508377batch_normalization_4_5508379batch_normalization_4_5508381batch_normalization_4_5508383*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55070162/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_55070572
re_lu_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_55061552!
max_pooling2d_4/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_15_5508388conv2d_15_5508390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_55070762#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_5508393batch_normalization_15_5508395batch_normalization_15_5508397batch_normalization_15_5508399*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_550712920
.batch_normalization_15/StatefulPartitionedCall?
re_lu_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_15_layer_call_and_return_conditional_losses_55071702
re_lu_15/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_15/PartitionedCall:output:0conv2d_34_5508403conv2d_34_5508405*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_55071882#
!conv2d_34/StatefulPartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_34_5508408batch_normalization_34_5508410batch_normalization_34_5508412batch_normalization_34_5508414*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_550724120
.batch_normalization_34/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_34_layer_call_and_return_conditional_losses_55072822
re_lu_34/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_55063752"
 max_pooling2d_34/PartitionedCall?
#conv2d_1592/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_1592_5508419conv2d_1592_5508421*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_55073012%
#conv2d_1592/StatefulPartitionedCall?
0batch_normalization_1592/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1592/StatefulPartitionedCall:output:0 batch_normalization_1592_5508424 batch_normalization_1592_5508426 batch_normalization_1592_5508428 batch_normalization_1592_5508430*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_550735422
0batch_normalization_1592/StatefulPartitionedCall?
re_lu_1484/PartitionedCallPartitionedCall9batch_normalization_1592/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_55073952
re_lu_1484/PartitionedCall?
"max_pooling2d_1484/PartitionedCallPartitionedCall#re_lu_1484/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_55064912$
"max_pooling2d_1484/PartitionedCall?
#conv2d_1598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0conv2d_1598_5508435conv2d_1598_5508437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_55074142%
#conv2d_1598/StatefulPartitionedCall?
0batch_normalization_1598/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1598/StatefulPartitionedCall:output:0 batch_normalization_1598_5508440 batch_normalization_1598_5508442 batch_normalization_1598_5508444 batch_normalization_1598_5508446*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_550746722
0batch_normalization_1598/StatefulPartitionedCall?
re_lu_1490/PartitionedCallPartitionedCall9batch_normalization_1598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_55075082
re_lu_1490/PartitionedCall?
"max_pooling2d_1490/PartitionedCallPartitionedCall#re_lu_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_55066072$
"max_pooling2d_1490/PartitionedCall?
add_765/PartitionedCallPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_765_layer_call_and_return_conditional_losses_55075232
add_765/PartitionedCall?
#conv2d_1608/StatefulPartitionedCallStatefulPartitionedCall add_765/PartitionedCall:output:0conv2d_1608_5508452conv2d_1608_5508454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_55075422%
#conv2d_1608/StatefulPartitionedCall?
0batch_normalization_1608/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1608/StatefulPartitionedCall:output:0 batch_normalization_1608_5508457 batch_normalization_1608_5508459 batch_normalization_1608_5508461 batch_normalization_1608_5508463*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_550759522
0batch_normalization_1608/StatefulPartitionedCall?
re_lu_1500/PartitionedCallPartitionedCall9batch_normalization_1608/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_55076362
re_lu_1500/PartitionedCall?
"max_pooling2d_1500/PartitionedCallPartitionedCall#re_lu_1500/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_55067232$
"max_pooling2d_1500/PartitionedCall?
add_766/PartitionedCallPartitionedCall+max_pooling2d_1500/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_766_layer_call_and_return_conditional_losses_55076512
add_766/PartitionedCall?
#conv2d_1612/StatefulPartitionedCallStatefulPartitionedCall add_766/PartitionedCall:output:0conv2d_1612_5508469conv2d_1612_5508471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_55076702%
#conv2d_1612/StatefulPartitionedCall?
0batch_normalization_1612/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1612/StatefulPartitionedCall:output:0 batch_normalization_1612_5508474 batch_normalization_1612_5508476 batch_normalization_1612_5508478 batch_normalization_1612_5508480*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_550772322
0batch_normalization_1612/StatefulPartitionedCall?
re_lu_1504/PartitionedCallPartitionedCall9batch_normalization_1612/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_55077642
re_lu_1504/PartitionedCall?
"max_pooling2d_1504/PartitionedCallPartitionedCall#re_lu_1504/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_55068392$
"max_pooling2d_1504/PartitionedCall?
add_767/PartitionedCallPartitionedCall+max_pooling2d_1504/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_767_layer_call_and_return_conditional_losses_55077792
add_767/PartitionedCall?
#conv2d_1625/StatefulPartitionedCallStatefulPartitionedCall add_767/PartitionedCall:output:0conv2d_1625_5508486conv2d_1625_5508488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_55077982%
#conv2d_1625/StatefulPartitionedCall?
0batch_normalization_1625/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1625/StatefulPartitionedCall:output:0 batch_normalization_1625_5508491 batch_normalization_1625_5508493 batch_normalization_1625_5508495 batch_normalization_1625_5508497*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_550785122
0batch_normalization_1625/StatefulPartitionedCall?
re_lu_1517/PartitionedCallPartitionedCall9batch_normalization_1625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_55078922
re_lu_1517/PartitionedCall?
flatten_439/PartitionedCallPartitionedCall#re_lu_1517/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_439_layer_call_and_return_conditional_losses_55079062
flatten_439/PartitionedCall?
!dense_878/StatefulPartitionedCallStatefulPartitionedCall$flatten_439/PartitionedCall:output:0dense_878_5508502dense_878_5508504*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_55079252#
!dense_878/StatefulPartitionedCall?
!dense_879/StatefulPartitionedCallStatefulPartitionedCall*dense_878/StatefulPartitionedCall:output:0dense_879_5508507dense_879_5508509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_879_layer_call_and_return_conditional_losses_55079522#
!dense_879/StatefulPartitionedCall?
IdentityIdentity*dense_879/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall1^batch_normalization_1592/StatefulPartitionedCall1^batch_normalization_1598/StatefulPartitionedCall1^batch_normalization_1608/StatefulPartitionedCall1^batch_normalization_1612/StatefulPartitionedCall1^batch_normalization_1625/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^conv2d_1592/StatefulPartitionedCall$^conv2d_1598/StatefulPartitionedCall$^conv2d_1608/StatefulPartitionedCall$^conv2d_1612/StatefulPartitionedCall$^conv2d_1625/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall"^dense_879/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2d
0batch_normalization_1592/StatefulPartitionedCall0batch_normalization_1592/StatefulPartitionedCall2d
0batch_normalization_1598/StatefulPartitionedCall0batch_normalization_1598/StatefulPartitionedCall2d
0batch_normalization_1608/StatefulPartitionedCall0batch_normalization_1608/StatefulPartitionedCall2d
0batch_normalization_1612/StatefulPartitionedCall0batch_normalization_1612/StatefulPartitionedCall2d
0batch_normalization_1625/StatefulPartitionedCall0batch_normalization_1625/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#conv2d_1592/StatefulPartitionedCall#conv2d_1592/StatefulPartitionedCall2J
#conv2d_1598/StatefulPartitionedCall#conv2d_1598/StatefulPartitionedCall2J
#conv2d_1608/StatefulPartitionedCall#conv2d_1608/StatefulPartitionedCall2J
#conv2d_1612/StatefulPartitionedCall#conv2d_1612/StatefulPartitionedCall2J
#conv2d_1625/StatefulPartitionedCall#conv2d_1625/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall2F
!dense_879/StatefulPartitionedCall!dense_879/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5507577

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5507851

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509893

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
p
D__inference_add_766_layer_call_and_return_conditional_losses_5510325
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????   2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
P
4__inference_max_pooling2d_1490_layer_call_fn_5506613

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_55066072
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
U
)__inference_add_766_layer_call_fn_5510331
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_766_layer_call_and_return_conditional_losses_55076512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510452

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_4_layer_call_fn_5509448

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55070162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1608_layer_call_fn_5510232

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_55066752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1612_layer_call_fn_5510465

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_55067912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5506938

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_4_layer_call_fn_5506161

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_55061552
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_15_layer_call_fn_5509592

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_55071112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509625

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509782

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1612_layer_call_fn_5510350

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_55076702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1598_layer_call_fn_5510140

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_55065902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
F__inference_dense_878_layer_call_and_return_conditional_losses_5507925

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5507241

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1625_layer_call_fn_5510647

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_55069382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5507449

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5507354

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_5506491

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5506138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5507595

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_5506839

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1598_layer_call_fn_5510127

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_55065592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
U
)__inference_add_767_layer_call_fn_5510500
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_767_layer_call_and_return_conditional_losses_55077792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????   :?????????   :Y U
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
P
4__inference_max_pooling2d_1500_layer_call_fn_5506729

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_55067232
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5507833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_model_438_layer_call_fn_5508620
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_438_layer_call_and_return_conditional_losses_55085132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
+__inference_model_438_layer_call_fn_5509365

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_438_layer_call_and_return_conditional_losses_55085132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510096

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
H
,__inference_re_lu_1500_layer_call_fn_5510319

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_55076362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5506822

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_5507414

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
H
,__inference_re_lu_1504_layer_call_fn_5510488

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_55077642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_34_layer_call_fn_5509813

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_55063272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509957

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_dense_878_layer_call_fn_5510688

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_55079252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1608_layer_call_fn_5510296

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_55075772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510265

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510370

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1612_layer_call_fn_5510401

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_55077052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_5507508

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_5506375

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1625_layer_call_fn_5510583

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_55078512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_conv2d_15_layer_call_fn_5509541

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_55070762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_5510483

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
H
,__inference_re_lu_1517_layer_call_fn_5510657

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_55078922
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????   :W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_model_438_layer_call_fn_5508367
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-.1234*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_438_layer_call_and_return_conditional_losses_55082602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5506107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5507076

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509579

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_5507057

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5506254

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5506706

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509561

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510283

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
-__inference_conv2d_1608_layer_call_fn_5510181

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_55075422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
??
?
F__inference_model_438_layer_call_and_return_conditional_losses_5508260

inputs
conv2d_4_5508119
conv2d_4_5508121!
batch_normalization_4_5508124!
batch_normalization_4_5508126!
batch_normalization_4_5508128!
batch_normalization_4_5508130
conv2d_15_5508135
conv2d_15_5508137"
batch_normalization_15_5508140"
batch_normalization_15_5508142"
batch_normalization_15_5508144"
batch_normalization_15_5508146
conv2d_34_5508150
conv2d_34_5508152"
batch_normalization_34_5508155"
batch_normalization_34_5508157"
batch_normalization_34_5508159"
batch_normalization_34_5508161
conv2d_1592_5508166
conv2d_1592_5508168$
 batch_normalization_1592_5508171$
 batch_normalization_1592_5508173$
 batch_normalization_1592_5508175$
 batch_normalization_1592_5508177
conv2d_1598_5508182
conv2d_1598_5508184$
 batch_normalization_1598_5508187$
 batch_normalization_1598_5508189$
 batch_normalization_1598_5508191$
 batch_normalization_1598_5508193
conv2d_1608_5508199
conv2d_1608_5508201$
 batch_normalization_1608_5508204$
 batch_normalization_1608_5508206$
 batch_normalization_1608_5508208$
 batch_normalization_1608_5508210
conv2d_1612_5508216
conv2d_1612_5508218$
 batch_normalization_1612_5508221$
 batch_normalization_1612_5508223$
 batch_normalization_1612_5508225$
 batch_normalization_1612_5508227
conv2d_1625_5508233
conv2d_1625_5508235$
 batch_normalization_1625_5508238$
 batch_normalization_1625_5508240$
 batch_normalization_1625_5508242$
 batch_normalization_1625_5508244
dense_878_5508249
dense_878_5508251
dense_879_5508254
dense_879_5508256
identity??.batch_normalization_15/StatefulPartitionedCall?0batch_normalization_1592/StatefulPartitionedCall?0batch_normalization_1598/StatefulPartitionedCall?0batch_normalization_1608/StatefulPartitionedCall?0batch_normalization_1612/StatefulPartitionedCall?0batch_normalization_1625/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#conv2d_1592/StatefulPartitionedCall?#conv2d_1598/StatefulPartitionedCall?#conv2d_1608/StatefulPartitionedCall?#conv2d_1612/StatefulPartitionedCall?#conv2d_1625/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?!dense_878/StatefulPartitionedCall?!dense_879/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_5508119conv2d_4_5508121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_55069632"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_5508124batch_normalization_4_5508126batch_normalization_4_5508128batch_normalization_4_5508130*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55069982/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_55070572
re_lu_4/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_55061552!
max_pooling2d_4/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_15_5508135conv2d_15_5508137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_55070762#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_5508140batch_normalization_15_5508142batch_normalization_15_5508144batch_normalization_15_5508146*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_550711120
.batch_normalization_15/StatefulPartitionedCall?
re_lu_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_15_layer_call_and_return_conditional_losses_55071702
re_lu_15/PartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_15/PartitionedCall:output:0conv2d_34_5508150conv2d_34_5508152*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_55071882#
!conv2d_34/StatefulPartitionedCall?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_34_5508155batch_normalization_34_5508157batch_normalization_34_5508159batch_normalization_34_5508161*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_550722320
.batch_normalization_34/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_34_layer_call_and_return_conditional_losses_55072822
re_lu_34/PartitionedCall?
 max_pooling2d_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_55063752"
 max_pooling2d_34/PartitionedCall?
#conv2d_1592/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_1592_5508166conv2d_1592_5508168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_55073012%
#conv2d_1592/StatefulPartitionedCall?
0batch_normalization_1592/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1592/StatefulPartitionedCall:output:0 batch_normalization_1592_5508171 batch_normalization_1592_5508173 batch_normalization_1592_5508175 batch_normalization_1592_5508177*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_550733622
0batch_normalization_1592/StatefulPartitionedCall?
re_lu_1484/PartitionedCallPartitionedCall9batch_normalization_1592/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_55073952
re_lu_1484/PartitionedCall?
"max_pooling2d_1484/PartitionedCallPartitionedCall#re_lu_1484/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_55064912$
"max_pooling2d_1484/PartitionedCall?
#conv2d_1598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0conv2d_1598_5508182conv2d_1598_5508184*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_55074142%
#conv2d_1598/StatefulPartitionedCall?
0batch_normalization_1598/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1598/StatefulPartitionedCall:output:0 batch_normalization_1598_5508187 batch_normalization_1598_5508189 batch_normalization_1598_5508191 batch_normalization_1598_5508193*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_550744922
0batch_normalization_1598/StatefulPartitionedCall?
re_lu_1490/PartitionedCallPartitionedCall9batch_normalization_1598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_55075082
re_lu_1490/PartitionedCall?
"max_pooling2d_1490/PartitionedCallPartitionedCall#re_lu_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_55066072$
"max_pooling2d_1490/PartitionedCall?
add_765/PartitionedCallPartitionedCall+max_pooling2d_1484/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_765_layer_call_and_return_conditional_losses_55075232
add_765/PartitionedCall?
#conv2d_1608/StatefulPartitionedCallStatefulPartitionedCall add_765/PartitionedCall:output:0conv2d_1608_5508199conv2d_1608_5508201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_55075422%
#conv2d_1608/StatefulPartitionedCall?
0batch_normalization_1608/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1608/StatefulPartitionedCall:output:0 batch_normalization_1608_5508204 batch_normalization_1608_5508206 batch_normalization_1608_5508208 batch_normalization_1608_5508210*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_550757722
0batch_normalization_1608/StatefulPartitionedCall?
re_lu_1500/PartitionedCallPartitionedCall9batch_normalization_1608/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_55076362
re_lu_1500/PartitionedCall?
"max_pooling2d_1500/PartitionedCallPartitionedCall#re_lu_1500/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_55067232$
"max_pooling2d_1500/PartitionedCall?
add_766/PartitionedCallPartitionedCall+max_pooling2d_1500/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_766_layer_call_and_return_conditional_losses_55076512
add_766/PartitionedCall?
#conv2d_1612/StatefulPartitionedCallStatefulPartitionedCall add_766/PartitionedCall:output:0conv2d_1612_5508216conv2d_1612_5508218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_55076702%
#conv2d_1612/StatefulPartitionedCall?
0batch_normalization_1612/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1612/StatefulPartitionedCall:output:0 batch_normalization_1612_5508221 batch_normalization_1612_5508223 batch_normalization_1612_5508225 batch_normalization_1612_5508227*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_550770522
0batch_normalization_1612/StatefulPartitionedCall?
re_lu_1504/PartitionedCallPartitionedCall9batch_normalization_1612/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_55077642
re_lu_1504/PartitionedCall?
"max_pooling2d_1504/PartitionedCallPartitionedCall#re_lu_1504/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_55068392$
"max_pooling2d_1504/PartitionedCall?
add_767/PartitionedCallPartitionedCall+max_pooling2d_1504/PartitionedCall:output:0+max_pooling2d_1490/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_767_layer_call_and_return_conditional_losses_55077792
add_767/PartitionedCall?
#conv2d_1625/StatefulPartitionedCallStatefulPartitionedCall add_767/PartitionedCall:output:0conv2d_1625_5508233conv2d_1625_5508235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_55077982%
#conv2d_1625/StatefulPartitionedCall?
0batch_normalization_1625/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1625/StatefulPartitionedCall:output:0 batch_normalization_1625_5508238 batch_normalization_1625_5508240 batch_normalization_1625_5508242 batch_normalization_1625_5508244*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_550783322
0batch_normalization_1625/StatefulPartitionedCall?
re_lu_1517/PartitionedCallPartitionedCall9batch_normalization_1625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_55078922
re_lu_1517/PartitionedCall?
flatten_439/PartitionedCallPartitionedCall#re_lu_1517/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_439_layer_call_and_return_conditional_losses_55079062
flatten_439/PartitionedCall?
!dense_878/StatefulPartitionedCallStatefulPartitionedCall$flatten_439/PartitionedCall:output:0dense_878_5508249dense_878_5508251*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_55079252#
!dense_878/StatefulPartitionedCall?
!dense_879/StatefulPartitionedCallStatefulPartitionedCall*dense_878/StatefulPartitionedCall:output:0dense_879_5508254dense_879_5508256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_879_layer_call_and_return_conditional_losses_55079522#
!dense_879/StatefulPartitionedCall?
IdentityIdentity*dense_879/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall1^batch_normalization_1592/StatefulPartitionedCall1^batch_normalization_1598/StatefulPartitionedCall1^batch_normalization_1608/StatefulPartitionedCall1^batch_normalization_1612/StatefulPartitionedCall1^batch_normalization_1625/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^conv2d_1592/StatefulPartitionedCall$^conv2d_1598/StatefulPartitionedCall$^conv2d_1608/StatefulPartitionedCall$^conv2d_1612/StatefulPartitionedCall$^conv2d_1625/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall"^dense_879/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2d
0batch_normalization_1592/StatefulPartitionedCall0batch_normalization_1592/StatefulPartitionedCall2d
0batch_normalization_1598/StatefulPartitionedCall0batch_normalization_1598/StatefulPartitionedCall2d
0batch_normalization_1608/StatefulPartitionedCall0batch_normalization_1608/StatefulPartitionedCall2d
0batch_normalization_1612/StatefulPartitionedCall0batch_normalization_1612/StatefulPartitionedCall2d
0batch_normalization_1625/StatefulPartitionedCall0batch_normalization_1625/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#conv2d_1592/StatefulPartitionedCall#conv2d_1592/StatefulPartitionedCall2J
#conv2d_1598/StatefulPartitionedCall#conv2d_1598/StatefulPartitionedCall2J
#conv2d_1608/StatefulPartitionedCall#conv2d_1608/StatefulPartitionedCall2J
#conv2d_1612/StatefulPartitionedCall#conv2d_1612/StatefulPartitionedCall2J
#conv2d_1625/StatefulPartitionedCall#conv2d_1625/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall2F
!dense_879/StatefulPartitionedCall!dense_879/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1608_layer_call_fn_5510245

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_55067062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????  =
	dense_8790
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??	
à
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-14
 layer-31
!layer_with_weights-15
!layer-32
"layer-33
#layer-34
$layer_with_weights-16
$layer-35
%layer_with_weights-17
%layer-36
&	optimizer
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model_438", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_438", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_15", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_15", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["re_lu_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_34", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_34", "inbound_nodes": [[["batch_normalization_34", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["re_lu_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1592", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1592", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1592", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1592", "inbound_nodes": [[["conv2d_1592", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1484", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1484", "inbound_nodes": [[["batch_normalization_1592", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1484", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1484", "inbound_nodes": [[["re_lu_1484", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1598", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1598", "inbound_nodes": [[["max_pooling2d_1484", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1598", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1598", "inbound_nodes": [[["conv2d_1598", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1490", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1490", "inbound_nodes": [[["batch_normalization_1598", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1490", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1490", "inbound_nodes": [[["re_lu_1490", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_765", "trainable": true, "dtype": "float32"}, "name": "add_765", "inbound_nodes": [[["max_pooling2d_1484", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1608", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1608", "inbound_nodes": [[["add_765", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1608", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1608", "inbound_nodes": [[["conv2d_1608", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1500", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1500", "inbound_nodes": [[["batch_normalization_1608", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1500", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1500", "inbound_nodes": [[["re_lu_1500", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_766", "trainable": true, "dtype": "float32"}, "name": "add_766", "inbound_nodes": [[["max_pooling2d_1500", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1612", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1612", "inbound_nodes": [[["add_766", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1612", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1612", "inbound_nodes": [[["conv2d_1612", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1504", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1504", "inbound_nodes": [[["batch_normalization_1612", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1504", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1504", "inbound_nodes": [[["re_lu_1504", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_767", "trainable": true, "dtype": "float32"}, "name": "add_767", "inbound_nodes": [[["max_pooling2d_1504", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1625", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1625", "inbound_nodes": [[["add_767", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1625", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1625", "inbound_nodes": [[["conv2d_1625", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1517", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1517", "inbound_nodes": [[["batch_normalization_1625", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_439", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_439", "inbound_nodes": [[["re_lu_1517", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_878", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_878", "inbound_nodes": [[["flatten_439", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_879", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_879", "inbound_nodes": [[["dense_878", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_879", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_438", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_15", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_15", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["re_lu_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_34", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_34", "inbound_nodes": [[["batch_normalization_34", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["re_lu_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1592", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1592", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1592", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1592", "inbound_nodes": [[["conv2d_1592", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1484", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1484", "inbound_nodes": [[["batch_normalization_1592", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1484", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1484", "inbound_nodes": [[["re_lu_1484", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1598", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1598", "inbound_nodes": [[["max_pooling2d_1484", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1598", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1598", "inbound_nodes": [[["conv2d_1598", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1490", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1490", "inbound_nodes": [[["batch_normalization_1598", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1490", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1490", "inbound_nodes": [[["re_lu_1490", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_765", "trainable": true, "dtype": "float32"}, "name": "add_765", "inbound_nodes": [[["max_pooling2d_1484", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1608", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1608", "inbound_nodes": [[["add_765", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1608", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1608", "inbound_nodes": [[["conv2d_1608", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1500", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1500", "inbound_nodes": [[["batch_normalization_1608", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1500", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1500", "inbound_nodes": [[["re_lu_1500", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_766", "trainable": true, "dtype": "float32"}, "name": "add_766", "inbound_nodes": [[["max_pooling2d_1500", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1612", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1612", "inbound_nodes": [[["add_766", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1612", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1612", "inbound_nodes": [[["conv2d_1612", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1504", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1504", "inbound_nodes": [[["batch_normalization_1612", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1504", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1504", "inbound_nodes": [[["re_lu_1504", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_767", "trainable": true, "dtype": "float32"}, "name": "add_767", "inbound_nodes": [[["max_pooling2d_1504", 0, 0, {}], ["max_pooling2d_1490", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1625", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1625", "inbound_nodes": [[["add_767", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1625", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1625", "inbound_nodes": [[["conv2d_1625", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1517", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1517", "inbound_nodes": [[["batch_normalization_1625", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_439", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_439", "inbound_nodes": [[["re_lu_1517", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_878", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_878", "inbound_nodes": [[["flatten_439", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_879", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_879", "inbound_nodes": [[["dense_878", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_879", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
?	
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_15", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_34", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1592", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1592", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1592", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1592", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
|regularization_losses
}trainable_variables
~	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1484", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1484", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1484", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1484", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1598", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1598", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1598", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1598", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1490", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1490", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1490", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1490", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_765", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_765", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 32]}, {"class_name": "TensorShape", "items": [null, 32, 32, 32]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1608", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1608", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1608", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1608", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1500", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1500", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1500", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1500", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_766", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_766", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 32]}, {"class_name": "TensorShape", "items": [null, 32, 32, 32]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1612", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1612", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1612", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1612", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1504", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1504", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1504", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1504", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_767", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_767", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 32]}, {"class_name": "TensorShape", "items": [null, 32, 32, 32]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1625", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1625", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1625", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1625", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1517", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1517", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_439", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_439", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_878", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_878", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32768]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_879", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_879", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter

?decay
?learning_rate
?momentum,momentum?-momentum?3momentum?4momentum?Cmomentum?Dmomentum?Jmomentum?Kmomentum?Vmomentum?Wmomentum?]momentum?^momentum?mmomentum?nmomentum?tmomentum?umomentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum?"
	optimizer
 "
trackable_list_wrapper
?
,0
-1
32
43
C4
D5
J6
K7
V8
W9
]10
^11
m12
n13
t14
u15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
?
,0
-1
32
43
54
65
C6
D7
J8
K9
L10
M11
V12
W13
]14
^15
_16
`17
m18
n19
t20
u21
v22
w23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
'regularization_losses
(trainable_variables
)	variables
 ?layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
.regularization_losses
/trainable_variables
0	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
7regularization_losses
8trainable_variables
9	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
;regularization_losses
<trainable_variables
=	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
@trainable_variables
A	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_15/kernel
:2conv2d_15/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_15/gamma
):'2batch_normalization_15/beta
2:0 (2"batch_normalization_15/moving_mean
6:4 (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_34/kernel
:2conv2d_34/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
Xregularization_losses
Ytrainable_variables
Z	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_34/gamma
):'2batch_normalization_34/beta
2:0 (2"batch_normalization_34/moving_mean
6:4 (2&batch_normalization_34/moving_variance
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
aregularization_losses
btrainable_variables
c	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
eregularization_losses
ftrainable_variables
g	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2conv2d_1592/kernel
: 2conv2d_1592/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
oregularization_losses
ptrainable_variables
q	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1592/gamma
+:) 2batch_normalization_1592/beta
4:2  (2$batch_normalization_1592/moving_mean
8:6  (2(batch_normalization_1592/moving_variance
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
xregularization_losses
ytrainable_variables
z	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
|regularization_losses
}trainable_variables
~	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1598/kernel
: 2conv2d_1598/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1598/gamma
+:) 2batch_normalization_1598/beta
4:2  (2$batch_normalization_1598/moving_mean
8:6  (2(batch_normalization_1598/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1608/kernel
: 2conv2d_1608/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1608/gamma
+:) 2batch_normalization_1608/beta
4:2  (2$batch_normalization_1608/moving_mean
8:6  (2(batch_normalization_1608/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1612/kernel
: 2conv2d_1612/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1612/gamma
+:) 2batch_normalization_1612/beta
4:2  (2$batch_normalization_1612/moving_mean
8:6  (2(batch_normalization_1612/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1625/kernel
: 2conv2d_1625/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1625/gamma
+:) 2batch_normalization_1625/beta
4:2  (2$batch_normalization_1625/moving_mean
8:6  (2(batch_normalization_1625/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#???2dense_878/kernel
:?2dense_878/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?
2dense_879/kernel
:
2dense_879/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?metrics
?non_trainable_variables
?regularization_losses
?trainable_variables
?	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
50
61
L2
M3
_4
`5
v6
w7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
4:22SGD/conv2d_4/kernel/momentum
&:$2SGD/conv2d_4/bias/momentum
4:22(SGD/batch_normalization_4/gamma/momentum
3:12'SGD/batch_normalization_4/beta/momentum
5:32SGD/conv2d_15/kernel/momentum
':%2SGD/conv2d_15/bias/momentum
5:32)SGD/batch_normalization_15/gamma/momentum
4:22(SGD/batch_normalization_15/beta/momentum
5:32SGD/conv2d_34/kernel/momentum
':%2SGD/conv2d_34/bias/momentum
5:32)SGD/batch_normalization_34/gamma/momentum
4:22(SGD/batch_normalization_34/beta/momentum
7:5 2SGD/conv2d_1592/kernel/momentum
):' 2SGD/conv2d_1592/bias/momentum
7:5 2+SGD/batch_normalization_1592/gamma/momentum
6:4 2*SGD/batch_normalization_1592/beta/momentum
7:5  2SGD/conv2d_1598/kernel/momentum
):' 2SGD/conv2d_1598/bias/momentum
7:5 2+SGD/batch_normalization_1598/gamma/momentum
6:4 2*SGD/batch_normalization_1598/beta/momentum
7:5  2SGD/conv2d_1608/kernel/momentum
):' 2SGD/conv2d_1608/bias/momentum
7:5 2+SGD/batch_normalization_1608/gamma/momentum
6:4 2*SGD/batch_normalization_1608/beta/momentum
7:5  2SGD/conv2d_1612/kernel/momentum
):' 2SGD/conv2d_1612/bias/momentum
7:5 2+SGD/batch_normalization_1612/gamma/momentum
6:4 2*SGD/batch_normalization_1612/beta/momentum
7:5  2SGD/conv2d_1625/kernel/momentum
):' 2SGD/conv2d_1625/bias/momentum
7:5 2+SGD/batch_normalization_1625/gamma/momentum
6:4 2*SGD/batch_normalization_1625/beta/momentum
0:.???2SGD/dense_878/kernel/momentum
(:&?2SGD/dense_878/bias/momentum
.:,	?
2SGD/dense_879/kernel/momentum
':%
2SGD/dense_879/bias/momentum
?2?
"__inference__wrapped_model_5506045?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????  
?2?
+__inference_model_438_layer_call_fn_5508620
+__inference_model_438_layer_call_fn_5509256
+__inference_model_438_layer_call_fn_5508367
+__inference_model_438_layer_call_fn_5509365?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_438_layer_call_and_return_conditional_losses_5508950
F__inference_model_438_layer_call_and_return_conditional_losses_5509147
F__inference_model_438_layer_call_and_return_conditional_losses_5508113
F__inference_model_438_layer_call_and_return_conditional_losses_5507969?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_4_layer_call_fn_5509384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5509375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_4_layer_call_fn_5509435
7__inference_batch_normalization_4_layer_call_fn_5509512
7__inference_batch_normalization_4_layer_call_fn_5509448
7__inference_batch_normalization_4_layer_call_fn_5509499?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509422
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509486
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509468
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509404?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_re_lu_4_layer_call_fn_5509522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_re_lu_4_layer_call_and_return_conditional_losses_5509517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_4_layer_call_fn_5506161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5506155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_15_layer_call_fn_5509541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5509532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_15_layer_call_fn_5509605
8__inference_batch_normalization_15_layer_call_fn_5509592
8__inference_batch_normalization_15_layer_call_fn_5509656
8__inference_batch_normalization_15_layer_call_fn_5509669?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509561
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509643
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509625
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509579?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_re_lu_15_layer_call_fn_5509679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_15_layer_call_and_return_conditional_losses_5509674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_34_layer_call_fn_5509698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_34_layer_call_and_return_conditional_losses_5509689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_34_layer_call_fn_5509749
8__inference_batch_normalization_34_layer_call_fn_5509762
8__inference_batch_normalization_34_layer_call_fn_5509826
8__inference_batch_normalization_34_layer_call_fn_5509813?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509782
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509736
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509800
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509718?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_re_lu_34_layer_call_fn_5509836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_34_layer_call_and_return_conditional_losses_5509831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_34_layer_call_fn_5506381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_5506375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_conv2d_1592_layer_call_fn_5509855?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_5509846?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_1592_layer_call_fn_5509906
:__inference_batch_normalization_1592_layer_call_fn_5509983
:__inference_batch_normalization_1592_layer_call_fn_5509970
:__inference_batch_normalization_1592_layer_call_fn_5509919?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509957
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509875
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509939
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509893?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_re_lu_1484_layer_call_fn_5509993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_5509988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_max_pooling2d_1484_layer_call_fn_5506497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_5506491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_conv2d_1598_layer_call_fn_5510012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_5510003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_1598_layer_call_fn_5510127
:__inference_batch_normalization_1598_layer_call_fn_5510140
:__inference_batch_normalization_1598_layer_call_fn_5510063
:__inference_batch_normalization_1598_layer_call_fn_5510076?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510032
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510096
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510114
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510050?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_re_lu_1490_layer_call_fn_5510150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_5510145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_max_pooling2d_1490_layer_call_fn_5506613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_5506607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_add_765_layer_call_fn_5510162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_add_765_layer_call_and_return_conditional_losses_5510156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_1608_layer_call_fn_5510181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_5510172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_1608_layer_call_fn_5510296
:__inference_batch_normalization_1608_layer_call_fn_5510309
:__inference_batch_normalization_1608_layer_call_fn_5510245
:__inference_batch_normalization_1608_layer_call_fn_5510232?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510283
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510265
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510201
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510219?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_re_lu_1500_layer_call_fn_5510319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_5510314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_max_pooling2d_1500_layer_call_fn_5506729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_5506723?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_add_766_layer_call_fn_5510331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_add_766_layer_call_and_return_conditional_losses_5510325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_1612_layer_call_fn_5510350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_5510341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_1612_layer_call_fn_5510414
:__inference_batch_normalization_1612_layer_call_fn_5510478
:__inference_batch_normalization_1612_layer_call_fn_5510465
:__inference_batch_normalization_1612_layer_call_fn_5510401?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510434
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510370
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510452
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510388?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_re_lu_1504_layer_call_fn_5510488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_5510483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_max_pooling2d_1504_layer_call_fn_5506845?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_5506839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_add_767_layer_call_fn_5510500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_add_767_layer_call_and_return_conditional_losses_5510494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_1625_layer_call_fn_5510519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_5510510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_1625_layer_call_fn_5510570
:__inference_batch_normalization_1625_layer_call_fn_5510583
:__inference_batch_normalization_1625_layer_call_fn_5510647
:__inference_batch_normalization_1625_layer_call_fn_5510634?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510621
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510539
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510603
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510557?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_re_lu_1517_layer_call_fn_5510657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_5510652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_439_layer_call_fn_5510668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_439_layer_call_and_return_conditional_losses_5510663?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_878_layer_call_fn_5510688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_878_layer_call_and_return_conditional_losses_5510679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_879_layer_call_fn_5510708?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_879_layer_call_and_return_conditional_losses_5510699?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_5508737input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_5506045?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????8?5
.?+
)?&
input_1?????????  
? "5?2
0
	dense_879#? 
	dense_879?????????
?
D__inference_add_765_layer_call_and_return_conditional_losses_5510156?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? "-?*
#? 
0?????????   
? ?
)__inference_add_765_layer_call_fn_5510162?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? " ??????????   ?
D__inference_add_766_layer_call_and_return_conditional_losses_5510325?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? "-?*
#? 
0?????????   
? ?
)__inference_add_766_layer_call_fn_5510331?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? " ??????????   ?
D__inference_add_767_layer_call_and_return_conditional_losses_5510494?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? "-?*
#? 
0?????????   
? ?
)__inference_add_767_layer_call_fn_5510500?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? " ??????????   ?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509875?tuvwM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509893?tuvwM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509939rtuvw;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1592_layer_call_and_return_conditional_losses_5509957rtuvw;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
:__inference_batch_normalization_1592_layer_call_fn_5509906?tuvwM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1592_layer_call_fn_5509919?tuvwM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1592_layer_call_fn_5509970etuvw;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1592_layer_call_fn_5509983etuvw;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510032v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510050v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510096?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1598_layer_call_and_return_conditional_losses_5510114?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
:__inference_batch_normalization_1598_layer_call_fn_5510063i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1598_layer_call_fn_5510076i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
:__inference_batch_normalization_1598_layer_call_fn_5510127?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1598_layer_call_fn_5510140?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509561rJKLM;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509579rJKLM;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509625?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5509643?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_15_layer_call_fn_5509592eJKLM;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
8__inference_batch_normalization_15_layer_call_fn_5509605eJKLM;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_15_layer_call_fn_5509656?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_15_layer_call_fn_5509669?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510201?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510219?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510265v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1608_layer_call_and_return_conditional_losses_5510283v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
:__inference_batch_normalization_1608_layer_call_fn_5510232?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1608_layer_call_fn_5510245?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1608_layer_call_fn_5510296i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1608_layer_call_fn_5510309i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510370v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510388v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510434?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1612_layer_call_and_return_conditional_losses_5510452?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
:__inference_batch_normalization_1612_layer_call_fn_5510401i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1612_layer_call_fn_5510414i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
:__inference_batch_normalization_1612_layer_call_fn_5510465?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1612_layer_call_fn_5510478?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510539v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510557v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510603?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1625_layer_call_and_return_conditional_losses_5510621?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
:__inference_batch_normalization_1625_layer_call_fn_5510570i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1625_layer_call_fn_5510583i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
:__inference_batch_normalization_1625_layer_call_fn_5510634?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1625_layer_call_fn_5510647?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509718r]^_`;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509736r]^_`;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509782?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_5509800?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_34_layer_call_fn_5509749e]^_`;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
8__inference_batch_normalization_34_layer_call_fn_5509762e]^_`;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_34_layer_call_fn_5509813?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_34_layer_call_fn_5509826?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509404r3456;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509422r3456;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509468?3456M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5509486?3456M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
7__inference_batch_normalization_4_layer_call_fn_5509435e3456;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
7__inference_batch_normalization_4_layer_call_fn_5509448e3456;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
7__inference_batch_normalization_4_layer_call_fn_5509499?3456M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_4_layer_call_fn_5509512?3456M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
H__inference_conv2d_1592_layer_call_and_return_conditional_losses_5509846lmn7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1592_layer_call_fn_5509855_mn7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
H__inference_conv2d_1598_layer_call_and_return_conditional_losses_5510003n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1598_layer_call_fn_5510012a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_5509532lCD7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_15_layer_call_fn_5509541_CD7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
H__inference_conv2d_1608_layer_call_and_return_conditional_losses_5510172n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1608_layer_call_fn_5510181a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
H__inference_conv2d_1612_layer_call_and_return_conditional_losses_5510341n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1612_layer_call_fn_5510350a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
H__inference_conv2d_1625_layer_call_and_return_conditional_losses_5510510n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1625_layer_call_fn_5510519a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
F__inference_conv2d_34_layer_call_and_return_conditional_losses_5509689lVW7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_34_layer_call_fn_5509698_VW7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_5509375l,-7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_4_layer_call_fn_5509384_,-7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
F__inference_dense_878_layer_call_and_return_conditional_losses_5510679a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ?
+__inference_dense_878_layer_call_fn_5510688T??1?.
'?$
"?
inputs???????????
? "????????????
F__inference_dense_879_layer_call_and_return_conditional_losses_5510699_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
+__inference_dense_879_layer_call_fn_5510708R??0?-
&?#
!?
inputs??????????
? "??????????
?
H__inference_flatten_439_layer_call_and_return_conditional_losses_5510663b7?4
-?*
(?%
inputs?????????   
? "'?$
?
0???????????
? ?
-__inference_flatten_439_layer_call_fn_5510668U7?4
-?*
(?%
inputs?????????   
? "?????????????
O__inference_max_pooling2d_1484_layer_call_and_return_conditional_losses_5506491?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1484_layer_call_fn_5506497?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_max_pooling2d_1490_layer_call_and_return_conditional_losses_5506607?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1490_layer_call_fn_5506613?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_max_pooling2d_1500_layer_call_and_return_conditional_losses_5506723?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1500_layer_call_fn_5506729?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_max_pooling2d_1504_layer_call_and_return_conditional_losses_5506839?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1504_layer_call_fn_5506845?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_5506375?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_34_layer_call_fn_5506381?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5506155?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_4_layer_call_fn_5506161?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_model_438_layer_call_and_return_conditional_losses_5507969?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????@?=
6?3
)?&
input_1?????????  
p

 
? "%?"
?
0?????????

? ?
F__inference_model_438_layer_call_and_return_conditional_losses_5508113?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????@?=
6?3
)?&
input_1?????????  
p 

 
? "%?"
?
0?????????

? ?
F__inference_model_438_layer_call_and_return_conditional_losses_5508950?P,-3456CDJKLMVW]^_`mntuvw??????????????????????????????<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
F__inference_model_438_layer_call_and_return_conditional_losses_5509147?P,-3456CDJKLMVW]^_`mntuvw??????????????????????????????<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
+__inference_model_438_layer_call_fn_5508367?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????@?=
6?3
)?&
input_1?????????  
p

 
? "??????????
?
+__inference_model_438_layer_call_fn_5508620?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????@?=
6?3
)?&
input_1?????????  
p 

 
? "??????????
?
+__inference_model_438_layer_call_fn_5509256?P,-3456CDJKLMVW]^_`mntuvw??????????????????????????????<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
+__inference_model_438_layer_call_fn_5509365?P,-3456CDJKLMVW]^_`mntuvw??????????????????????????????<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
G__inference_re_lu_1484_layer_call_and_return_conditional_losses_5509988h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1484_layer_call_fn_5509993[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1490_layer_call_and_return_conditional_losses_5510145h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1490_layer_call_fn_5510150[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1500_layer_call_and_return_conditional_losses_5510314h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1500_layer_call_fn_5510319[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1504_layer_call_and_return_conditional_losses_5510483h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1504_layer_call_fn_5510488[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1517_layer_call_and_return_conditional_losses_5510652h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1517_layer_call_fn_5510657[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
E__inference_re_lu_15_layer_call_and_return_conditional_losses_5509674h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_re_lu_15_layer_call_fn_5509679[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_re_lu_34_layer_call_and_return_conditional_losses_5509831h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_re_lu_34_layer_call_fn_5509836[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
D__inference_re_lu_4_layer_call_and_return_conditional_losses_5509517h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
)__inference_re_lu_4_layer_call_fn_5509522[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
%__inference_signature_wrapper_5508737?P,-3456CDJKLMVW]^_`mntuvw????????????????????????????C?@
? 
9?6
4
input_1)?&
input_1?????????  "5?2
0
	dense_879#? 
	dense_879?????????
