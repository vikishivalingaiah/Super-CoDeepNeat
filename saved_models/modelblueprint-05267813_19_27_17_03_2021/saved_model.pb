??5
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??,
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:*
dtype0
?
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_35/gamma
?
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_35/beta
?
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_35/moving_mean
?
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_35/moving_variance
?
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1936/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv2d_1936/kernel
?
&conv2d_1936/kernel/Read/ReadVariableOpReadVariableOpconv2d_1936/kernel*&
_output_shapes
: *
dtype0
x
conv2d_1936/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1936/bias
q
$conv2d_1936/bias/Read/ReadVariableOpReadVariableOpconv2d_1936/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1936/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1936/gamma
?
2batch_normalization_1936/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1936/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1936/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1936/beta
?
1batch_normalization_1936/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1936/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1936/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1936/moving_mean
?
8batch_normalization_1936/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1936/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1936/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1936/moving_variance
?
<batch_normalization_1936/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1936/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1968/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1968/kernel
?
&conv2d_1968/kernel/Read/ReadVariableOpReadVariableOpconv2d_1968/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1968/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1968/bias
q
$conv2d_1968/bias/Read/ReadVariableOpReadVariableOpconv2d_1968/bias*
_output_shapes
: *
dtype0
?
batch_normalization_1968/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1968/gamma
?
2batch_normalization_1968/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1968/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_1968/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1968/beta
?
1batch_normalization_1968/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1968/beta*
_output_shapes
: *
dtype0
?
$batch_normalization_1968/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1968/moving_mean
?
8batch_normalization_1968/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1968/moving_mean*
_output_shapes
: *
dtype0
?
(batch_normalization_1968/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1968/moving_variance
?
<batch_normalization_1968/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1968/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_163/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_163/kernel

%conv2d_163/kernel/Read/ReadVariableOpReadVariableOpconv2d_163/kernel*&
_output_shapes
: *
dtype0
v
conv2d_163/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_163/bias
o
#conv2d_163/bias/Read/ReadVariableOpReadVariableOpconv2d_163/bias*
_output_shapes
: *
dtype0
?
batch_normalization_163/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_163/gamma
?
1batch_normalization_163/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_163/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_163/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_163/beta
?
0batch_normalization_163/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_163/beta*
_output_shapes
: *
dtype0
?
#batch_normalization_163/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_163/moving_mean
?
7batch_normalization_163/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_163/moving_mean*
_output_shapes
: *
dtype0
?
'batch_normalization_163/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_163/moving_variance
?
;batch_normalization_163/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_163/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_195/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *"
shared_nameconv2d_195/kernel

%conv2d_195/kernel/Read/ReadVariableOpReadVariableOpconv2d_195/kernel*&
_output_shapes
:  *
dtype0
v
conv2d_195/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_195/bias
o
#conv2d_195/bias/Read/ReadVariableOpReadVariableOpconv2d_195/bias*
_output_shapes
: *
dtype0
?
batch_normalization_195/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_195/gamma
?
1batch_normalization_195/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_195/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_195/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_195/beta
?
0batch_normalization_195/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_195/beta*
_output_shapes
: *
dtype0
?
#batch_normalization_195/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_195/moving_mean
?
7batch_normalization_195/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_195/moving_mean*
_output_shapes
: *
dtype0
?
'batch_normalization_195/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_195/moving_variance
?
;batch_normalization_195/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_195/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1522/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv2d_1522/kernel
?
&conv2d_1522/kernel/Read/ReadVariableOpReadVariableOpconv2d_1522/kernel*&
_output_shapes
: @*
dtype0
x
conv2d_1522/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_1522/bias
q
$conv2d_1522/bias/Read/ReadVariableOpReadVariableOpconv2d_1522/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_1522/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name batch_normalization_1522/gamma
?
2batch_normalization_1522/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1522/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1522/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_1522/beta
?
1batch_normalization_1522/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1522/beta*
_output_shapes
:@*
dtype0
?
$batch_normalization_1522/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_normalization_1522/moving_mean
?
8batch_normalization_1522/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1522/moving_mean*
_output_shapes
:@*
dtype0
?
(batch_normalization_1522/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(batch_normalization_1522/moving_variance
?
<batch_normalization_1522/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1522/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_1556/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv2d_1556/kernel
?
&conv2d_1556/kernel/Read/ReadVariableOpReadVariableOpconv2d_1556/kernel*&
_output_shapes
:@@*
dtype0
x
conv2d_1556/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_1556/bias
q
$conv2d_1556/bias/Read/ReadVariableOpReadVariableOpconv2d_1556/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_1556/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name batch_normalization_1556/gamma
?
2batch_normalization_1556/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1556/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1556/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_1556/beta
?
1batch_normalization_1556/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1556/beta*
_output_shapes
:@*
dtype0
?
$batch_normalization_1556/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_normalization_1556/moving_mean
?
8batch_normalization_1556/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1556/moving_mean*
_output_shapes
:@*
dtype0
?
(batch_normalization_1556/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(batch_normalization_1556/moving_variance
?
<batch_normalization_1556/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1556/moving_variance*
_output_shapes
:@*
dtype0

dense_630/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*!
shared_namedense_630/kernel
x
$dense_630/kernel/Read/ReadVariableOpReadVariableOpdense_630/kernel*!
_output_shapes
:???*
dtype0
u
dense_630/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_630/bias
n
"dense_630/bias/Read/ReadVariableOpReadVariableOpdense_630/bias*
_output_shapes	
:?*
dtype0
}
dense_631/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_namedense_631/kernel
v
$dense_631/kernel/Read/ReadVariableOpReadVariableOpdense_631/kernel*
_output_shapes
:	?
*
dtype0
t
dense_631/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_631/bias
m
"dense_631/bias/Read/ReadVariableOpReadVariableOpdense_631/bias*
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
?
+conv_adjust_channels_235/conv2d_2155/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+conv_adjust_channels_235/conv2d_2155/kernel
?
?conv_adjust_channels_235/conv2d_2155/kernel/Read/ReadVariableOpReadVariableOp+conv_adjust_channels_235/conv2d_2155/kernel*&
_output_shapes
: *
dtype0
?
)conv_adjust_channels_235/conv2d_2155/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)conv_adjust_channels_235/conv2d_2155/bias
?
=conv_adjust_channels_235/conv2d_2155/bias/Read/ReadVariableOpReadVariableOp)conv_adjust_channels_235/conv2d_2155/bias*
_output_shapes
:*
dtype0
?
7conv_adjust_channels_235/batch_normalization_2155/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97conv_adjust_channels_235/batch_normalization_2155/gamma
?
Kconv_adjust_channels_235/batch_normalization_2155/gamma/Read/ReadVariableOpReadVariableOp7conv_adjust_channels_235/batch_normalization_2155/gamma*
_output_shapes
:*
dtype0
?
6conv_adjust_channels_235/batch_normalization_2155/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86conv_adjust_channels_235/batch_normalization_2155/beta
?
Jconv_adjust_channels_235/batch_normalization_2155/beta/Read/ReadVariableOpReadVariableOp6conv_adjust_channels_235/batch_normalization_2155/beta*
_output_shapes
:*
dtype0
?
=conv_adjust_channels_235/batch_normalization_2155/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=conv_adjust_channels_235/batch_normalization_2155/moving_mean
?
Qconv_adjust_channels_235/batch_normalization_2155/moving_mean/Read/ReadVariableOpReadVariableOp=conv_adjust_channels_235/batch_normalization_2155/moving_mean*
_output_shapes
:*
dtype0
?
Aconv_adjust_channels_235/batch_normalization_2155/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAconv_adjust_channels_235/batch_normalization_2155/moving_variance
?
Uconv_adjust_channels_235/batch_normalization_2155/moving_variance/Read/ReadVariableOpReadVariableOpAconv_adjust_channels_235/batch_normalization_2155/moving_variance*
_output_shapes
:*
dtype0
?
SGD/conv2d_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameSGD/conv2d_6/kernel/momentum
?
0SGD/conv2d_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_6/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv2d_6/bias/momentum
?
.SGD/conv2d_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_6/bias/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_6/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_6/gamma/momentum
?
<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_6/gamma/momentum*
_output_shapes
:*
dtype0
?
'SGD/batch_normalization_6/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'SGD/batch_normalization_6/beta/momentum
?
;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_6/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_27/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_27/kernel/momentum
?
1SGD/conv2d_27/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_27/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_27/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameSGD/conv2d_27/bias/momentum
?
/SGD/conv2d_27/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_27/bias/momentum*
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_27/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_27/gamma/momentum
?
=SGD/batch_normalization_27/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_27/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_27/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_27/beta/momentum
?
<SGD/batch_normalization_27/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_27/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_35/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_35/kernel/momentum
?
1SGD/conv2d_35/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_35/kernel/momentum*&
_output_shapes
:*
dtype0
?
SGD/conv2d_35/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameSGD/conv2d_35/bias/momentum
?
/SGD/conv2d_35/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_35/bias/momentum*
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_35/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_35/gamma/momentum
?
=SGD/batch_normalization_35/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_35/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_35/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_35/beta/momentum
?
<SGD/batch_normalization_35/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_35/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/conv2d_1936/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!SGD/conv2d_1936/kernel/momentum
?
3SGD/conv2d_1936/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1936/kernel/momentum*&
_output_shapes
: *
dtype0
?
SGD/conv2d_1936/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1936/bias/momentum
?
1SGD/conv2d_1936/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1936/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1936/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1936/gamma/momentum
?
?SGD/batch_normalization_1936/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1936/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1936/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1936/beta/momentum
?
>SGD/batch_normalization_1936/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1936/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1968/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!SGD/conv2d_1968/kernel/momentum
?
3SGD/conv2d_1968/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1968/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_1968/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/conv2d_1968/bias/momentum
?
1SGD/conv2d_1968/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1968/bias/momentum*
_output_shapes
: *
dtype0
?
+SGD/batch_normalization_1968/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+SGD/batch_normalization_1968/gamma/momentum
?
?SGD/batch_normalization_1968/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1968/gamma/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_1968/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_1968/beta/momentum
?
>SGD/batch_normalization_1968/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1968/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_163/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name SGD/conv2d_163/kernel/momentum
?
2SGD/conv2d_163/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_163/kernel/momentum*&
_output_shapes
: *
dtype0
?
SGD/conv2d_163/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameSGD/conv2d_163/bias/momentum
?
0SGD/conv2d_163/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_163/bias/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_163/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_163/gamma/momentum
?
>SGD/batch_normalization_163/gamma/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_163/gamma/momentum*
_output_shapes
: *
dtype0
?
)SGD/batch_normalization_163/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_163/beta/momentum
?
=SGD/batch_normalization_163/beta/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_163/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_195/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  */
shared_name SGD/conv2d_195/kernel/momentum
?
2SGD/conv2d_195/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_195/kernel/momentum*&
_output_shapes
:  *
dtype0
?
SGD/conv2d_195/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameSGD/conv2d_195/bias/momentum
?
0SGD/conv2d_195/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_195/bias/momentum*
_output_shapes
: *
dtype0
?
*SGD/batch_normalization_195/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*SGD/batch_normalization_195/gamma/momentum
?
>SGD/batch_normalization_195/gamma/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_195/gamma/momentum*
_output_shapes
: *
dtype0
?
)SGD/batch_normalization_195/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_195/beta/momentum
?
=SGD/batch_normalization_195/beta/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_195/beta/momentum*
_output_shapes
: *
dtype0
?
SGD/conv2d_1522/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*0
shared_name!SGD/conv2d_1522/kernel/momentum
?
3SGD/conv2d_1522/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1522/kernel/momentum*&
_output_shapes
: @*
dtype0
?
SGD/conv2d_1522/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameSGD/conv2d_1522/bias/momentum
?
1SGD/conv2d_1522/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1522/bias/momentum*
_output_shapes
:@*
dtype0
?
+SGD/batch_normalization_1522/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/batch_normalization_1522/gamma/momentum
?
?SGD/batch_normalization_1522/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1522/gamma/momentum*
_output_shapes
:@*
dtype0
?
*SGD/batch_normalization_1522/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*SGD/batch_normalization_1522/beta/momentum
?
>SGD/batch_normalization_1522/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1522/beta/momentum*
_output_shapes
:@*
dtype0
?
SGD/conv2d_1556/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!SGD/conv2d_1556/kernel/momentum
?
3SGD/conv2d_1556/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1556/kernel/momentum*&
_output_shapes
:@@*
dtype0
?
SGD/conv2d_1556/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameSGD/conv2d_1556/bias/momentum
?
1SGD/conv2d_1556/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_1556/bias/momentum*
_output_shapes
:@*
dtype0
?
+SGD/batch_normalization_1556/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+SGD/batch_normalization_1556/gamma/momentum
?
?SGD/batch_normalization_1556/gamma/momentum/Read/ReadVariableOpReadVariableOp+SGD/batch_normalization_1556/gamma/momentum*
_output_shapes
:@*
dtype0
?
*SGD/batch_normalization_1556/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*SGD/batch_normalization_1556/beta/momentum
?
>SGD/batch_normalization_1556/beta/momentum/Read/ReadVariableOpReadVariableOp*SGD/batch_normalization_1556/beta/momentum*
_output_shapes
:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?	
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
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer-25
layer_with_weights-15
layer-26
layer_with_weights-16
layer-27
layer-28
layer-29
layer_with_weights-17
layer-30
 layer_with_weights-18
 layer-31
!layer-32
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer_with_weights-20
%layer-36
&	optimizer
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+
signatures
 
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
?
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
R
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
h

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
atrainable_variables
bregularization_losses
c	variables
d	keras_api
R
etrainable_variables
fregularization_losses
g	variables
h	keras_api
h

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
?
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
j

|kernel
}bias
~trainable_variables
regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
j
	?conv
?bn
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?iter

?decay
?learning_rate
?momentum,momentum?-momentum?3momentum?4momentum?Cmomentum?Dmomentum?Jmomentum?Kmomentum?Vmomentum?Wmomentum?]momentum?^momentum?imomentum?jmomentum?pmomentum?qmomentum?|momentum?}momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum?
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
i12
j13
p14
q15
|16
}17
?18
?19
?20
?21
?22
?23
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
 
?
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
i18
j19
p20
q21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
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
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63
?
 ?layer_regularization_losses
'trainable_variables
?layer_metrics
(regularization_losses
?layers
?non_trainable_variables
)	variables
?metrics
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
 ?layer_regularization_losses
.trainable_variables
?layer_metrics
?layers
/regularization_losses
?non_trainable_variables
0	variables
?metrics
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
52
63
?
 ?layer_regularization_losses
7trainable_variables
?layer_metrics
?layers
8regularization_losses
?non_trainable_variables
9	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
;trainable_variables
?layer_metrics
?layers
<regularization_losses
?non_trainable_variables
=	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
@regularization_losses
?non_trainable_variables
A	variables
?metrics
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
?
 ?layer_regularization_losses
Etrainable_variables
?layer_metrics
?layers
Fregularization_losses
?non_trainable_variables
G	variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_27/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_27/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_27/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_27/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
L2
M3
?
 ?layer_regularization_losses
Ntrainable_variables
?layer_metrics
?layers
Oregularization_losses
?non_trainable_variables
P	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
Rtrainable_variables
?layer_metrics
?layers
Sregularization_losses
?non_trainable_variables
T	variables
?metrics
\Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
?
 ?layer_regularization_losses
Xtrainable_variables
?layer_metrics
?layers
Yregularization_losses
?non_trainable_variables
Z	variables
?metrics
 
ge
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
_2
`3
?
 ?layer_regularization_losses
atrainable_variables
?layer_metrics
?layers
bregularization_losses
?non_trainable_variables
c	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
etrainable_variables
?layer_metrics
?layers
fregularization_losses
?non_trainable_variables
g	variables
?metrics
^\
VARIABLE_VALUEconv2d_1936/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_1936/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
?
 ?layer_regularization_losses
ktrainable_variables
?layer_metrics
?layers
lregularization_losses
?non_trainable_variables
m	variables
?metrics
 
ig
VARIABLE_VALUEbatch_normalization_1936/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_1936/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_1936/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_1936/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
r2
s3
?
 ?layer_regularization_losses
ttrainable_variables
?layer_metrics
?layers
uregularization_losses
?non_trainable_variables
v	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
xtrainable_variables
?layer_metrics
?layers
yregularization_losses
?non_trainable_variables
z	variables
?metrics
^\
VARIABLE_VALUEconv2d_1968/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_1968/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
 

|0
}1
?
 ?layer_regularization_losses
~trainable_variables
?layer_metrics
?layers
regularization_losses
?non_trainable_variables
?	variables
?metrics
 
ig
VARIABLE_VALUEbatch_normalization_1968/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_1968/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_1968/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_1968/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
?0
?1
?2
?3
 
0
?0
?1
?2
?3
?4
?5
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
^\
VARIABLE_VALUEconv2d_163/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_163/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
ig
VARIABLE_VALUEbatch_normalization_163/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_163/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_163/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_163/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
^\
VARIABLE_VALUEconv2d_195/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_195/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
ig
VARIABLE_VALUEbatch_normalization_195/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_195/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_195/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_195/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
_]
VARIABLE_VALUEconv2d_1522/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_1522/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
jh
VARIABLE_VALUEbatch_normalization_1522/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_1522/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_1522/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_1522/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
_]
VARIABLE_VALUEconv2d_1556/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_1556/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
jh
VARIABLE_VALUEbatch_normalization_1556/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_1556/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_1556/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_1556/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
][
VARIABLE_VALUEdense_630/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_630/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
][
VARIABLE_VALUEdense_631/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_631/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+conv_adjust_channels_235/conv2d_2155/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE)conv_adjust_channels_235/conv2d_2155/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7conv_adjust_channels_235/batch_normalization_2155/gamma1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE6conv_adjust_channels_235/batch_normalization_2155/beta1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=conv_adjust_channels_235/batch_normalization_2155/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAconv_adjust_channels_235/batch_normalization_2155/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
 
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
?
50
61
L2
M3
_4
`5
r6
s7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
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

r0
s1
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

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 

?0
?1
 
 
?0
?1
?2
?3
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
 
 

?0
?1

?0
?1
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
?0
?1
 
??
VARIABLE_VALUESGD/conv2d_6/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_6/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_6/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'SGD/batch_normalization_6/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_27/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_27/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_27/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_27/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_35/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_35/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_35/gamma/momentumXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_35/beta/momentumWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1936/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1936/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1936/gamma/momentumXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1936/beta/momentumWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1968/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1968/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1968/gamma/momentumXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1968/beta/momentumWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_163/kernel/momentumZlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_163/bias/momentumXlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_163/gamma/momentumYlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_163/beta/momentumXlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_195/kernel/momentumZlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_195/bias/momentumXlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_195/gamma/momentumYlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_195/beta/momentumXlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1522/kernel/momentumZlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1522/bias/momentumXlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1522/gamma/momentumYlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1522/beta/momentumXlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1556/kernel/momentumZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/conv2d_1556/bias/momentumXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+SGD/batch_normalization_1556/gamma/momentumYlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/batch_normalization_1556/beta/momentumXlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_varianceconv2d_1936/kernelconv2d_1936/biasbatch_normalization_1936/gammabatch_normalization_1936/beta$batch_normalization_1936/moving_mean(batch_normalization_1936/moving_varianceconv2d_1968/kernelconv2d_1968/biasbatch_normalization_1968/gammabatch_normalization_1968/beta$batch_normalization_1968/moving_mean(batch_normalization_1968/moving_variance+conv_adjust_channels_235/conv2d_2155/kernel)conv_adjust_channels_235/conv2d_2155/bias7conv_adjust_channels_235/batch_normalization_2155/gamma6conv_adjust_channels_235/batch_normalization_2155/beta=conv_adjust_channels_235/batch_normalization_2155/moving_meanAconv_adjust_channels_235/batch_normalization_2155/moving_varianceconv2d_163/kernelconv2d_163/biasbatch_normalization_163/gammabatch_normalization_163/beta#batch_normalization_163/moving_mean'batch_normalization_163/moving_varianceconv2d_195/kernelconv2d_195/biasbatch_normalization_195/gammabatch_normalization_195/beta#batch_normalization_195/moving_mean'batch_normalization_195/moving_varianceconv2d_1522/kernelconv2d_1522/biasbatch_normalization_1522/gammabatch_normalization_1522/beta$batch_normalization_1522/moving_mean(batch_normalization_1522/moving_varianceconv2d_1556/kernelconv2d_1556/biasbatch_normalization_1556/gammabatch_normalization_1556/beta$batch_normalization_1556/moving_mean(batch_normalization_1556/moving_variancedense_630/kerneldense_630/biasdense_631/kerneldense_631/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_7475729
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp&conv2d_1936/kernel/Read/ReadVariableOp$conv2d_1936/bias/Read/ReadVariableOp2batch_normalization_1936/gamma/Read/ReadVariableOp1batch_normalization_1936/beta/Read/ReadVariableOp8batch_normalization_1936/moving_mean/Read/ReadVariableOp<batch_normalization_1936/moving_variance/Read/ReadVariableOp&conv2d_1968/kernel/Read/ReadVariableOp$conv2d_1968/bias/Read/ReadVariableOp2batch_normalization_1968/gamma/Read/ReadVariableOp1batch_normalization_1968/beta/Read/ReadVariableOp8batch_normalization_1968/moving_mean/Read/ReadVariableOp<batch_normalization_1968/moving_variance/Read/ReadVariableOp%conv2d_163/kernel/Read/ReadVariableOp#conv2d_163/bias/Read/ReadVariableOp1batch_normalization_163/gamma/Read/ReadVariableOp0batch_normalization_163/beta/Read/ReadVariableOp7batch_normalization_163/moving_mean/Read/ReadVariableOp;batch_normalization_163/moving_variance/Read/ReadVariableOp%conv2d_195/kernel/Read/ReadVariableOp#conv2d_195/bias/Read/ReadVariableOp1batch_normalization_195/gamma/Read/ReadVariableOp0batch_normalization_195/beta/Read/ReadVariableOp7batch_normalization_195/moving_mean/Read/ReadVariableOp;batch_normalization_195/moving_variance/Read/ReadVariableOp&conv2d_1522/kernel/Read/ReadVariableOp$conv2d_1522/bias/Read/ReadVariableOp2batch_normalization_1522/gamma/Read/ReadVariableOp1batch_normalization_1522/beta/Read/ReadVariableOp8batch_normalization_1522/moving_mean/Read/ReadVariableOp<batch_normalization_1522/moving_variance/Read/ReadVariableOp&conv2d_1556/kernel/Read/ReadVariableOp$conv2d_1556/bias/Read/ReadVariableOp2batch_normalization_1556/gamma/Read/ReadVariableOp1batch_normalization_1556/beta/Read/ReadVariableOp8batch_normalization_1556/moving_mean/Read/ReadVariableOp<batch_normalization_1556/moving_variance/Read/ReadVariableOp$dense_630/kernel/Read/ReadVariableOp"dense_630/bias/Read/ReadVariableOp$dense_631/kernel/Read/ReadVariableOp"dense_631/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp?conv_adjust_channels_235/conv2d_2155/kernel/Read/ReadVariableOp=conv_adjust_channels_235/conv2d_2155/bias/Read/ReadVariableOpKconv_adjust_channels_235/batch_normalization_2155/gamma/Read/ReadVariableOpJconv_adjust_channels_235/batch_normalization_2155/beta/Read/ReadVariableOpQconv_adjust_channels_235/batch_normalization_2155/moving_mean/Read/ReadVariableOpUconv_adjust_channels_235/batch_normalization_2155/moving_variance/Read/ReadVariableOp0SGD/conv2d_6/kernel/momentum/Read/ReadVariableOp.SGD/conv2d_6/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOp1SGD/conv2d_27/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_27/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_27/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_27/beta/momentum/Read/ReadVariableOp1SGD/conv2d_35/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_35/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_35/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_35/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1936/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1936/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1936/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1936/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1968/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1968/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1968/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1968/beta/momentum/Read/ReadVariableOp2SGD/conv2d_163/kernel/momentum/Read/ReadVariableOp0SGD/conv2d_163/bias/momentum/Read/ReadVariableOp>SGD/batch_normalization_163/gamma/momentum/Read/ReadVariableOp=SGD/batch_normalization_163/beta/momentum/Read/ReadVariableOp2SGD/conv2d_195/kernel/momentum/Read/ReadVariableOp0SGD/conv2d_195/bias/momentum/Read/ReadVariableOp>SGD/batch_normalization_195/gamma/momentum/Read/ReadVariableOp=SGD/batch_normalization_195/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1522/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1522/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1522/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1522/beta/momentum/Read/ReadVariableOp3SGD/conv2d_1556/kernel/momentum/Read/ReadVariableOp1SGD/conv2d_1556/bias/momentum/Read/ReadVariableOp?SGD/batch_normalization_1556/gamma/momentum/Read/ReadVariableOp>SGD/batch_normalization_1556/beta/momentum/Read/ReadVariableOpConst*u
Tinn
l2j	*
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
 __inference__traced_save_7478537
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_varianceconv2d_1936/kernelconv2d_1936/biasbatch_normalization_1936/gammabatch_normalization_1936/beta$batch_normalization_1936/moving_mean(batch_normalization_1936/moving_varianceconv2d_1968/kernelconv2d_1968/biasbatch_normalization_1968/gammabatch_normalization_1968/beta$batch_normalization_1968/moving_mean(batch_normalization_1968/moving_varianceconv2d_163/kernelconv2d_163/biasbatch_normalization_163/gammabatch_normalization_163/beta#batch_normalization_163/moving_mean'batch_normalization_163/moving_varianceconv2d_195/kernelconv2d_195/biasbatch_normalization_195/gammabatch_normalization_195/beta#batch_normalization_195/moving_mean'batch_normalization_195/moving_varianceconv2d_1522/kernelconv2d_1522/biasbatch_normalization_1522/gammabatch_normalization_1522/beta$batch_normalization_1522/moving_mean(batch_normalization_1522/moving_varianceconv2d_1556/kernelconv2d_1556/biasbatch_normalization_1556/gammabatch_normalization_1556/beta$batch_normalization_1556/moving_mean(batch_normalization_1556/moving_variancedense_630/kerneldense_630/biasdense_631/kerneldense_631/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentum+conv_adjust_channels_235/conv2d_2155/kernel)conv_adjust_channels_235/conv2d_2155/bias7conv_adjust_channels_235/batch_normalization_2155/gamma6conv_adjust_channels_235/batch_normalization_2155/beta=conv_adjust_channels_235/batch_normalization_2155/moving_meanAconv_adjust_channels_235/batch_normalization_2155/moving_varianceSGD/conv2d_6/kernel/momentumSGD/conv2d_6/bias/momentum(SGD/batch_normalization_6/gamma/momentum'SGD/batch_normalization_6/beta/momentumSGD/conv2d_27/kernel/momentumSGD/conv2d_27/bias/momentum)SGD/batch_normalization_27/gamma/momentum(SGD/batch_normalization_27/beta/momentumSGD/conv2d_35/kernel/momentumSGD/conv2d_35/bias/momentum)SGD/batch_normalization_35/gamma/momentum(SGD/batch_normalization_35/beta/momentumSGD/conv2d_1936/kernel/momentumSGD/conv2d_1936/bias/momentum+SGD/batch_normalization_1936/gamma/momentum*SGD/batch_normalization_1936/beta/momentumSGD/conv2d_1968/kernel/momentumSGD/conv2d_1968/bias/momentum+SGD/batch_normalization_1968/gamma/momentum*SGD/batch_normalization_1968/beta/momentumSGD/conv2d_163/kernel/momentumSGD/conv2d_163/bias/momentum*SGD/batch_normalization_163/gamma/momentum)SGD/batch_normalization_163/beta/momentumSGD/conv2d_195/kernel/momentumSGD/conv2d_195/bias/momentum*SGD/batch_normalization_195/gamma/momentum)SGD/batch_normalization_195/beta/momentumSGD/conv2d_1522/kernel/momentumSGD/conv2d_1522/bias/momentum+SGD/batch_normalization_1522/gamma/momentum*SGD/batch_normalization_1522/beta/momentumSGD/conv2d_1556/kernel/momentumSGD/conv2d_1556/bias/momentum+SGD/batch_normalization_1556/gamma/momentum*SGD/batch_normalization_1556/beta/momentum*t
Tinm
k2i*
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
#__inference__traced_restore_7478859??(
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7473705

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7474172

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477214

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
?
?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7473309

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
-__inference_conv2d_1522_layer_call_fn_7477709

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
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_74745322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
+__inference_conv2d_35_layer_call_fn_7476816

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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_74738952
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
?
?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7473486

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476761

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
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477578

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476586

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
?
?
8__inference_batch_normalization_27_layer_call_fn_7476723

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_74738362
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
?
?
7__inference_batch_normalization_6_layer_call_fn_7476617

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737052
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
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7473382

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
k
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_7473650

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
9__inference_batch_normalization_195_layer_call_fn_7477591

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_74744402
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
?
F
*__inference_re_lu_35_layer_call_fn_7476954

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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_74739892
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
?	
?
F__inference_dense_631_layer_call_and_return_conditional_losses_7478046

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7472773

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7472877

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
?
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473145
input_1
conv2d_2155_7473130
conv2d_2155_7473132$
 batch_normalization_2155_7473135$
 batch_normalization_2155_7473137$
 batch_normalization_2155_7473139$
 batch_normalization_2155_7473141
identity??0batch_normalization_2155/StatefulPartitionedCall?#conv2d_2155/StatefulPartitionedCall?
#conv2d_2155/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_2155_7473130conv2d_2155_7473132*
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
GPU 2J 8? *Q
fLRJ
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_74730372%
#conv2d_2155/StatefulPartitionedCall?
0batch_normalization_2155/StatefulPartitionedCallStatefulPartitionedCall,conv2d_2155/StatefulPartitionedCall:output:0 batch_normalization_2155_7473135 batch_normalization_2155_7473137 batch_normalization_2155_7473139 batch_normalization_2155_7473141*
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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_747309022
0batch_normalization_2155/StatefulPartitionedCall?
IdentityIdentity9batch_normalization_2155/StatefulPartitionedCall:output:01^batch_normalization_2155/StatefulPartitionedCall$^conv2d_2155/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2d
0batch_normalization_2155/StatefulPartitionedCall0batch_normalization_2155/StatefulPartitionedCall2J
#conv2d_2155/StatefulPartitionedCall#conv2d_2155/StatefulPartitionedCall:X T
/
_output_shapes
:?????????   
!
_user_specified_name	input_1
?
?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477811

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1522_layer_call_fn_7477760

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
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_74734862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477747

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1936_layer_call_fn_7477037

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_74740602
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
?
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473201

inputs
conv2d_2155_7473186
conv2d_2155_7473188$
 batch_normalization_2155_7473191$
 batch_normalization_2155_7473193$
 batch_normalization_2155_7473195$
 batch_normalization_2155_7473197
identity??0batch_normalization_2155/StatefulPartitionedCall?#conv2d_2155/StatefulPartitionedCall?
#conv2d_2155/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2155_7473186conv2d_2155_7473188*
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
GPU 2J 8? *Q
fLRJ
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_74730372%
#conv2d_2155/StatefulPartitionedCall?
0batch_normalization_2155/StatefulPartitionedCallStatefulPartitionedCall,conv2d_2155/StatefulPartitionedCall:output:0 batch_normalization_2155_7473191 batch_normalization_2155_7473193 batch_normalization_2155_7473195 batch_normalization_2155_7473197*
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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_747309022
0batch_normalization_2155/StatefulPartitionedCall?
IdentityIdentity9batch_normalization_2155/StatefulPartitionedCall:output:01^batch_normalization_2155/StatefulPartitionedCall$^conv2d_2155/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2d
0batch_normalization_2155/StatefulPartitionedCall0batch_normalization_2155/StatefulPartitionedCall2J
#conv2d_2155/StatefulPartitionedCall#conv2d_2155/StatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7472565

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
:__inference_batch_normalization_2155_layer_call_fn_7478138

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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_74730902
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
?
b
F__inference_re_lu_163_layer_call_and_return_conditional_losses_7477516

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
G__inference_conv2d_163_layer_call_and_return_conditional_losses_7474293

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
?
?
7__inference_batch_normalization_6_layer_call_fn_7476553

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74724492
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
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477950

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477421

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
?
?
:__inference_conv_adjust_channels_235_layer_call_fn_7473216
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_74732012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????   
!
_user_specified_name	input_1
?
G
+__inference_re_lu_195_layer_call_fn_7477678

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
GPU 2J 8? *O
fJRH
F__inference_re_lu_195_layer_call_and_return_conditional_losses_74744992
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
:__inference_batch_normalization_1968_layer_call_fn_7477181

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_74728772
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
?
?
7__inference_batch_normalization_6_layer_call_fn_7476630

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737232
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
?
p
D__inference_add_357_layer_call_and_return_conditional_losses_7477684
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
?
+__inference_dense_630_layer_call_fn_7478035

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
F__inference_dense_630_layer_call_and_return_conditional_losses_74747732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476900

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
?	
?
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_7474532

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7472700

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
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477729

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_7474213

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
-__inference_conv2d_1556_layer_call_fn_7477866

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
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_74746452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
b
F__inference_re_lu_163_layer_call_and_return_conditional_losses_7474387

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
:__inference_batch_normalization_1936_layer_call_fn_7477101

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_74728042
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
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7473517

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_35_layer_call_fn_7476931

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_74739302
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
?
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_7472497

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
+__inference_conv2d_27_layer_call_fn_7476659

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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_74737832
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
??
?
F__inference_model_314_layer_call_and_return_conditional_losses_7475457

inputs
conv2d_6_7475292
conv2d_6_7475294!
batch_normalization_6_7475297!
batch_normalization_6_7475299!
batch_normalization_6_7475301!
batch_normalization_6_7475303
conv2d_27_7475308
conv2d_27_7475310"
batch_normalization_27_7475313"
batch_normalization_27_7475315"
batch_normalization_27_7475317"
batch_normalization_27_7475319
conv2d_35_7475323
conv2d_35_7475325"
batch_normalization_35_7475328"
batch_normalization_35_7475330"
batch_normalization_35_7475332"
batch_normalization_35_7475334
conv2d_1936_7475338
conv2d_1936_7475340$
 batch_normalization_1936_7475343$
 batch_normalization_1936_7475345$
 batch_normalization_1936_7475347$
 batch_normalization_1936_7475349
conv2d_1968_7475353
conv2d_1968_7475355$
 batch_normalization_1968_7475358$
 batch_normalization_1968_7475360$
 batch_normalization_1968_7475362$
 batch_normalization_1968_7475364$
 conv_adjust_channels_235_7475368$
 conv_adjust_channels_235_7475370$
 conv_adjust_channels_235_7475372$
 conv_adjust_channels_235_7475374$
 conv_adjust_channels_235_7475376$
 conv_adjust_channels_235_7475378
conv2d_163_7475382
conv2d_163_7475384#
batch_normalization_163_7475387#
batch_normalization_163_7475389#
batch_normalization_163_7475391#
batch_normalization_163_7475393
conv2d_195_7475397
conv2d_195_7475399#
batch_normalization_195_7475402#
batch_normalization_195_7475404#
batch_normalization_195_7475406#
batch_normalization_195_7475408
conv2d_1522_7475413
conv2d_1522_7475415$
 batch_normalization_1522_7475418$
 batch_normalization_1522_7475420$
 batch_normalization_1522_7475422$
 batch_normalization_1522_7475424
conv2d_1556_7475429
conv2d_1556_7475431$
 batch_normalization_1556_7475434$
 batch_normalization_1556_7475436$
 batch_normalization_1556_7475438$
 batch_normalization_1556_7475440
dense_630_7475446
dense_630_7475448
dense_631_7475451
dense_631_7475453
identity??0batch_normalization_1522/StatefulPartitionedCall?0batch_normalization_1556/StatefulPartitionedCall?/batch_normalization_163/StatefulPartitionedCall?0batch_normalization_1936/StatefulPartitionedCall?/batch_normalization_195/StatefulPartitionedCall?0batch_normalization_1968/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?#conv2d_1522/StatefulPartitionedCall?#conv2d_1556/StatefulPartitionedCall?"conv2d_163/StatefulPartitionedCall?#conv2d_1936/StatefulPartitionedCall?"conv2d_195/StatefulPartitionedCall?#conv2d_1968/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?0conv_adjust_channels_235/StatefulPartitionedCall?!dense_630/StatefulPartitionedCall?!dense_631/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_7475292conv2d_6_7475294*
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_74736702"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_7475297batch_normalization_6_7475299batch_normalization_6_7475301batch_normalization_6_7475303*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737232/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_74737642
re_lu_6/PartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_74724972!
max_pooling2d_6/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_27_7475308conv2d_27_7475310*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_74737832#
!conv2d_27/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_27_7475313batch_normalization_27_7475315batch_normalization_27_7475317batch_normalization_27_7475319*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_747383620
.batch_normalization_27/StatefulPartitionedCall?
re_lu_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_74738772
re_lu_27/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall!re_lu_27/PartitionedCall:output:0conv2d_35_7475323conv2d_35_7475325*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_74738952#
!conv2d_35/StatefulPartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_7475328batch_normalization_35_7475330batch_normalization_35_7475332batch_normalization_35_7475334*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_747394820
.batch_normalization_35/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_74739892
re_lu_35/PartitionedCall?
#conv2d_1936/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0conv2d_1936_7475338conv2d_1936_7475340*
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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_74740072%
#conv2d_1936/StatefulPartitionedCall?
0batch_normalization_1936/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1936/StatefulPartitionedCall:output:0 batch_normalization_1936_7475343 batch_normalization_1936_7475345 batch_normalization_1936_7475347 batch_normalization_1936_7475349*
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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_747406022
0batch_normalization_1936/StatefulPartitionedCall?
re_lu_1763/PartitionedCallPartitionedCall9batch_normalization_1936/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_74741012
re_lu_1763/PartitionedCall?
#conv2d_1968/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1763/PartitionedCall:output:0conv2d_1968_7475353conv2d_1968_7475355*
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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_74741192%
#conv2d_1968/StatefulPartitionedCall?
0batch_normalization_1968/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1968/StatefulPartitionedCall:output:0 batch_normalization_1968_7475358 batch_normalization_1968_7475360 batch_normalization_1968_7475362 batch_normalization_1968_7475364*
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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_747417222
0batch_normalization_1968/StatefulPartitionedCall?
re_lu_1795/PartitionedCallPartitionedCall9batch_normalization_1968/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_74742132
re_lu_1795/PartitionedCall?
0conv_adjust_channels_235/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1795/PartitionedCall:output:0 conv_adjust_channels_235_7475368 conv_adjust_channels_235_7475370 conv_adjust_channels_235_7475372 conv_adjust_channels_235_7475374 conv_adjust_channels_235_7475376 conv_adjust_channels_235_7475378*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_747320122
0conv_adjust_channels_235/StatefulPartitionedCall?
add_356/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:09conv_adjust_channels_235/StatefulPartitionedCall:output:0*
Tin
2*
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
D__inference_add_356_layer_call_and_return_conditional_losses_74742742
add_356/PartitionedCall?
"conv2d_163/StatefulPartitionedCallStatefulPartitionedCall add_356/PartitionedCall:output:0conv2d_163_7475382conv2d_163_7475384*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_163_layer_call_and_return_conditional_losses_74742932$
"conv2d_163/StatefulPartitionedCall?
/batch_normalization_163/StatefulPartitionedCallStatefulPartitionedCall+conv2d_163/StatefulPartitionedCall:output:0batch_normalization_163_7475387batch_normalization_163_7475389batch_normalization_163_7475391batch_normalization_163_7475393*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_747434621
/batch_normalization_163/StatefulPartitionedCall?
re_lu_163/PartitionedCallPartitionedCall8batch_normalization_163/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_163_layer_call_and_return_conditional_losses_74743872
re_lu_163/PartitionedCall?
"conv2d_195/StatefulPartitionedCallStatefulPartitionedCall"re_lu_163/PartitionedCall:output:0conv2d_195_7475397conv2d_195_7475399*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_195_layer_call_and_return_conditional_losses_74744052$
"conv2d_195/StatefulPartitionedCall?
/batch_normalization_195/StatefulPartitionedCallStatefulPartitionedCall+conv2d_195/StatefulPartitionedCall:output:0batch_normalization_195_7475402batch_normalization_195_7475404batch_normalization_195_7475406batch_normalization_195_7475408*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_747445821
/batch_normalization_195/StatefulPartitionedCall?
re_lu_195/PartitionedCallPartitionedCall8batch_normalization_195/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_195_layer_call_and_return_conditional_losses_74744992
re_lu_195/PartitionedCall?
add_357/PartitionedCallPartitionedCall"re_lu_195/PartitionedCall:output:0#re_lu_1795/PartitionedCall:output:0*
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
D__inference_add_357_layer_call_and_return_conditional_losses_74745132
add_357/PartitionedCall?
#conv2d_1522/StatefulPartitionedCallStatefulPartitionedCall add_357/PartitionedCall:output:0conv2d_1522_7475413conv2d_1522_7475415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_74745322%
#conv2d_1522/StatefulPartitionedCall?
0batch_normalization_1522/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1522/StatefulPartitionedCall:output:0 batch_normalization_1522_7475418 batch_normalization_1522_7475420 batch_normalization_1522_7475422 batch_normalization_1522_7475424*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_747458522
0batch_normalization_1522/StatefulPartitionedCall?
re_lu_1404/PartitionedCallPartitionedCall9batch_normalization_1522/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_74746262
re_lu_1404/PartitionedCall?
"max_pooling2d_1404/PartitionedCallPartitionedCall#re_lu_1404/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_74735342$
"max_pooling2d_1404/PartitionedCall?
#conv2d_1556/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1404/PartitionedCall:output:0conv2d_1556_7475429conv2d_1556_7475431*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_74746452%
#conv2d_1556/StatefulPartitionedCall?
0batch_normalization_1556/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1556/StatefulPartitionedCall:output:0 batch_normalization_1556_7475434 batch_normalization_1556_7475436 batch_normalization_1556_7475438 batch_normalization_1556_7475440*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_747469822
0batch_normalization_1556/StatefulPartitionedCall?
re_lu_1438/PartitionedCallPartitionedCall9batch_normalization_1556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_74747392
re_lu_1438/PartitionedCall?
"max_pooling2d_1438/PartitionedCallPartitionedCall#re_lu_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_74736502$
"max_pooling2d_1438/PartitionedCall?
flatten_315/PartitionedCallPartitionedCall+max_pooling2d_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_315_layer_call_and_return_conditional_losses_74747542
flatten_315/PartitionedCall?
!dense_630/StatefulPartitionedCallStatefulPartitionedCall$flatten_315/PartitionedCall:output:0dense_630_7475446dense_630_7475448*
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
F__inference_dense_630_layer_call_and_return_conditional_losses_74747732#
!dense_630/StatefulPartitionedCall?
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_7475451dense_631_7475453*
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
F__inference_dense_631_layer_call_and_return_conditional_losses_74748002#
!dense_631/StatefulPartitionedCall?
IdentityIdentity*dense_631/StatefulPartitionedCall:output:01^batch_normalization_1522/StatefulPartitionedCall1^batch_normalization_1556/StatefulPartitionedCall0^batch_normalization_163/StatefulPartitionedCall1^batch_normalization_1936/StatefulPartitionedCall0^batch_normalization_195/StatefulPartitionedCall1^batch_normalization_1968/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall$^conv2d_1522/StatefulPartitionedCall$^conv2d_1556/StatefulPartitionedCall#^conv2d_163/StatefulPartitionedCall$^conv2d_1936/StatefulPartitionedCall#^conv2d_195/StatefulPartitionedCall$^conv2d_1968/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall1^conv_adjust_channels_235/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2d
0batch_normalization_1522/StatefulPartitionedCall0batch_normalization_1522/StatefulPartitionedCall2d
0batch_normalization_1556/StatefulPartitionedCall0batch_normalization_1556/StatefulPartitionedCall2b
/batch_normalization_163/StatefulPartitionedCall/batch_normalization_163/StatefulPartitionedCall2d
0batch_normalization_1936/StatefulPartitionedCall0batch_normalization_1936/StatefulPartitionedCall2b
/batch_normalization_195/StatefulPartitionedCall/batch_normalization_195/StatefulPartitionedCall2d
0batch_normalization_1968/StatefulPartitionedCall0batch_normalization_1968/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2J
#conv2d_1522/StatefulPartitionedCall#conv2d_1522/StatefulPartitionedCall2J
#conv2d_1556/StatefulPartitionedCall#conv2d_1556/StatefulPartitionedCall2H
"conv2d_163/StatefulPartitionedCall"conv2d_163/StatefulPartitionedCall2J
#conv2d_1936/StatefulPartitionedCall#conv2d_1936/StatefulPartitionedCall2H
"conv2d_195/StatefulPartitionedCall"conv2d_195/StatefulPartitionedCall2J
#conv2d_1968/StatefulPartitionedCall#conv2d_1968/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2d
0conv_adjust_channels_235/StatefulPartitionedCall0conv_adjust_channels_235/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
F__inference_model_314_layer_call_and_return_conditional_losses_7474985
input_1
conv2d_6_7474820
conv2d_6_7474822!
batch_normalization_6_7474825!
batch_normalization_6_7474827!
batch_normalization_6_7474829!
batch_normalization_6_7474831
conv2d_27_7474836
conv2d_27_7474838"
batch_normalization_27_7474841"
batch_normalization_27_7474843"
batch_normalization_27_7474845"
batch_normalization_27_7474847
conv2d_35_7474851
conv2d_35_7474853"
batch_normalization_35_7474856"
batch_normalization_35_7474858"
batch_normalization_35_7474860"
batch_normalization_35_7474862
conv2d_1936_7474866
conv2d_1936_7474868$
 batch_normalization_1936_7474871$
 batch_normalization_1936_7474873$
 batch_normalization_1936_7474875$
 batch_normalization_1936_7474877
conv2d_1968_7474881
conv2d_1968_7474883$
 batch_normalization_1968_7474886$
 batch_normalization_1968_7474888$
 batch_normalization_1968_7474890$
 batch_normalization_1968_7474892$
 conv_adjust_channels_235_7474896$
 conv_adjust_channels_235_7474898$
 conv_adjust_channels_235_7474900$
 conv_adjust_channels_235_7474902$
 conv_adjust_channels_235_7474904$
 conv_adjust_channels_235_7474906
conv2d_163_7474910
conv2d_163_7474912#
batch_normalization_163_7474915#
batch_normalization_163_7474917#
batch_normalization_163_7474919#
batch_normalization_163_7474921
conv2d_195_7474925
conv2d_195_7474927#
batch_normalization_195_7474930#
batch_normalization_195_7474932#
batch_normalization_195_7474934#
batch_normalization_195_7474936
conv2d_1522_7474941
conv2d_1522_7474943$
 batch_normalization_1522_7474946$
 batch_normalization_1522_7474948$
 batch_normalization_1522_7474950$
 batch_normalization_1522_7474952
conv2d_1556_7474957
conv2d_1556_7474959$
 batch_normalization_1556_7474962$
 batch_normalization_1556_7474964$
 batch_normalization_1556_7474966$
 batch_normalization_1556_7474968
dense_630_7474974
dense_630_7474976
dense_631_7474979
dense_631_7474981
identity??0batch_normalization_1522/StatefulPartitionedCall?0batch_normalization_1556/StatefulPartitionedCall?/batch_normalization_163/StatefulPartitionedCall?0batch_normalization_1936/StatefulPartitionedCall?/batch_normalization_195/StatefulPartitionedCall?0batch_normalization_1968/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?#conv2d_1522/StatefulPartitionedCall?#conv2d_1556/StatefulPartitionedCall?"conv2d_163/StatefulPartitionedCall?#conv2d_1936/StatefulPartitionedCall?"conv2d_195/StatefulPartitionedCall?#conv2d_1968/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?0conv_adjust_channels_235/StatefulPartitionedCall?!dense_630/StatefulPartitionedCall?!dense_631/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_6_7474820conv2d_6_7474822*
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_74736702"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_7474825batch_normalization_6_7474827batch_normalization_6_7474829batch_normalization_6_7474831*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737232/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_74737642
re_lu_6/PartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_74724972!
max_pooling2d_6/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_27_7474836conv2d_27_7474838*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_74737832#
!conv2d_27/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_27_7474841batch_normalization_27_7474843batch_normalization_27_7474845batch_normalization_27_7474847*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_747383620
.batch_normalization_27/StatefulPartitionedCall?
re_lu_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_74738772
re_lu_27/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall!re_lu_27/PartitionedCall:output:0conv2d_35_7474851conv2d_35_7474853*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_74738952#
!conv2d_35/StatefulPartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_7474856batch_normalization_35_7474858batch_normalization_35_7474860batch_normalization_35_7474862*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_747394820
.batch_normalization_35/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_74739892
re_lu_35/PartitionedCall?
#conv2d_1936/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0conv2d_1936_7474866conv2d_1936_7474868*
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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_74740072%
#conv2d_1936/StatefulPartitionedCall?
0batch_normalization_1936/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1936/StatefulPartitionedCall:output:0 batch_normalization_1936_7474871 batch_normalization_1936_7474873 batch_normalization_1936_7474875 batch_normalization_1936_7474877*
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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_747406022
0batch_normalization_1936/StatefulPartitionedCall?
re_lu_1763/PartitionedCallPartitionedCall9batch_normalization_1936/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_74741012
re_lu_1763/PartitionedCall?
#conv2d_1968/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1763/PartitionedCall:output:0conv2d_1968_7474881conv2d_1968_7474883*
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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_74741192%
#conv2d_1968/StatefulPartitionedCall?
0batch_normalization_1968/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1968/StatefulPartitionedCall:output:0 batch_normalization_1968_7474886 batch_normalization_1968_7474888 batch_normalization_1968_7474890 batch_normalization_1968_7474892*
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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_747417222
0batch_normalization_1968/StatefulPartitionedCall?
re_lu_1795/PartitionedCallPartitionedCall9batch_normalization_1968/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_74742132
re_lu_1795/PartitionedCall?
0conv_adjust_channels_235/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1795/PartitionedCall:output:0 conv_adjust_channels_235_7474896 conv_adjust_channels_235_7474898 conv_adjust_channels_235_7474900 conv_adjust_channels_235_7474902 conv_adjust_channels_235_7474904 conv_adjust_channels_235_7474906*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_747320122
0conv_adjust_channels_235/StatefulPartitionedCall?
add_356/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:09conv_adjust_channels_235/StatefulPartitionedCall:output:0*
Tin
2*
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
D__inference_add_356_layer_call_and_return_conditional_losses_74742742
add_356/PartitionedCall?
"conv2d_163/StatefulPartitionedCallStatefulPartitionedCall add_356/PartitionedCall:output:0conv2d_163_7474910conv2d_163_7474912*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_163_layer_call_and_return_conditional_losses_74742932$
"conv2d_163/StatefulPartitionedCall?
/batch_normalization_163/StatefulPartitionedCallStatefulPartitionedCall+conv2d_163/StatefulPartitionedCall:output:0batch_normalization_163_7474915batch_normalization_163_7474917batch_normalization_163_7474919batch_normalization_163_7474921*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_747434621
/batch_normalization_163/StatefulPartitionedCall?
re_lu_163/PartitionedCallPartitionedCall8batch_normalization_163/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_163_layer_call_and_return_conditional_losses_74743872
re_lu_163/PartitionedCall?
"conv2d_195/StatefulPartitionedCallStatefulPartitionedCall"re_lu_163/PartitionedCall:output:0conv2d_195_7474925conv2d_195_7474927*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_195_layer_call_and_return_conditional_losses_74744052$
"conv2d_195/StatefulPartitionedCall?
/batch_normalization_195/StatefulPartitionedCallStatefulPartitionedCall+conv2d_195/StatefulPartitionedCall:output:0batch_normalization_195_7474930batch_normalization_195_7474932batch_normalization_195_7474934batch_normalization_195_7474936*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_747445821
/batch_normalization_195/StatefulPartitionedCall?
re_lu_195/PartitionedCallPartitionedCall8batch_normalization_195/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_195_layer_call_and_return_conditional_losses_74744992
re_lu_195/PartitionedCall?
add_357/PartitionedCallPartitionedCall"re_lu_195/PartitionedCall:output:0#re_lu_1795/PartitionedCall:output:0*
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
D__inference_add_357_layer_call_and_return_conditional_losses_74745132
add_357/PartitionedCall?
#conv2d_1522/StatefulPartitionedCallStatefulPartitionedCall add_357/PartitionedCall:output:0conv2d_1522_7474941conv2d_1522_7474943*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_74745322%
#conv2d_1522/StatefulPartitionedCall?
0batch_normalization_1522/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1522/StatefulPartitionedCall:output:0 batch_normalization_1522_7474946 batch_normalization_1522_7474948 batch_normalization_1522_7474950 batch_normalization_1522_7474952*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_747458522
0batch_normalization_1522/StatefulPartitionedCall?
re_lu_1404/PartitionedCallPartitionedCall9batch_normalization_1522/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_74746262
re_lu_1404/PartitionedCall?
"max_pooling2d_1404/PartitionedCallPartitionedCall#re_lu_1404/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_74735342$
"max_pooling2d_1404/PartitionedCall?
#conv2d_1556/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1404/PartitionedCall:output:0conv2d_1556_7474957conv2d_1556_7474959*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_74746452%
#conv2d_1556/StatefulPartitionedCall?
0batch_normalization_1556/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1556/StatefulPartitionedCall:output:0 batch_normalization_1556_7474962 batch_normalization_1556_7474964 batch_normalization_1556_7474966 batch_normalization_1556_7474968*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_747469822
0batch_normalization_1556/StatefulPartitionedCall?
re_lu_1438/PartitionedCallPartitionedCall9batch_normalization_1556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_74747392
re_lu_1438/PartitionedCall?
"max_pooling2d_1438/PartitionedCallPartitionedCall#re_lu_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_74736502$
"max_pooling2d_1438/PartitionedCall?
flatten_315/PartitionedCallPartitionedCall+max_pooling2d_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_315_layer_call_and_return_conditional_losses_74747542
flatten_315/PartitionedCall?
!dense_630/StatefulPartitionedCallStatefulPartitionedCall$flatten_315/PartitionedCall:output:0dense_630_7474974dense_630_7474976*
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
F__inference_dense_630_layer_call_and_return_conditional_losses_74747732#
!dense_630/StatefulPartitionedCall?
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_7474979dense_631_7474981*
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
F__inference_dense_631_layer_call_and_return_conditional_losses_74748002#
!dense_631/StatefulPartitionedCall?
IdentityIdentity*dense_631/StatefulPartitionedCall:output:01^batch_normalization_1522/StatefulPartitionedCall1^batch_normalization_1556/StatefulPartitionedCall0^batch_normalization_163/StatefulPartitionedCall1^batch_normalization_1936/StatefulPartitionedCall0^batch_normalization_195/StatefulPartitionedCall1^batch_normalization_1968/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall$^conv2d_1522/StatefulPartitionedCall$^conv2d_1556/StatefulPartitionedCall#^conv2d_163/StatefulPartitionedCall$^conv2d_1936/StatefulPartitionedCall#^conv2d_195/StatefulPartitionedCall$^conv2d_1968/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall1^conv_adjust_channels_235/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2d
0batch_normalization_1522/StatefulPartitionedCall0batch_normalization_1522/StatefulPartitionedCall2d
0batch_normalization_1556/StatefulPartitionedCall0batch_normalization_1556/StatefulPartitionedCall2b
/batch_normalization_163/StatefulPartitionedCall/batch_normalization_163/StatefulPartitionedCall2d
0batch_normalization_1936/StatefulPartitionedCall0batch_normalization_1936/StatefulPartitionedCall2b
/batch_normalization_195/StatefulPartitionedCall/batch_normalization_195/StatefulPartitionedCall2d
0batch_normalization_1968/StatefulPartitionedCall0batch_normalization_1968/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2J
#conv2d_1522/StatefulPartitionedCall#conv2d_1522/StatefulPartitionedCall2J
#conv2d_1556/StatefulPartitionedCall#conv2d_1556/StatefulPartitionedCall2H
"conv2d_163/StatefulPartitionedCall"conv2d_163/StatefulPartitionedCall2J
#conv2d_1936/StatefulPartitionedCall#conv2d_1936/StatefulPartitionedCall2H
"conv2d_195/StatefulPartitionedCall"conv2d_195/StatefulPartitionedCall2J
#conv2d_1968/StatefulPartitionedCall#conv2d_1968/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2d
0conv_adjust_channels_235/StatefulPartitionedCall0conv_adjust_channels_235/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_7476635

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
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478094

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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_7474101

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
?
?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7474585

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
b
F__inference_re_lu_195_layer_call_and_return_conditional_losses_7474499

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
F__inference_dense_630_layer_call_and_return_conditional_losses_7474773

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_35_layer_call_and_return_conditional_losses_7476807

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
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476604

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
H
,__inference_re_lu_1404_layer_call_fn_7477847

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
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_74746262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7473278

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
?
?
+__inference_model_314_layer_call_fn_7475287
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

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity??StatefulPartitionedCall?	
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"%&'(+,-.1234789:=>?@*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_314_layer_call_and_return_conditional_losses_74751562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
c
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_7474626

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7472981

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
I
-__inference_flatten_315_layer_call_fn_7478015

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
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_315_layer_call_and_return_conditional_losses_74747542
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7474042

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
??
?6
F__inference_model_314_layer_call_and_return_conditional_losses_7476217

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource2
.batch_normalization_35_readvariableop_resource4
0batch_normalization_35_readvariableop_1_resourceC
?batch_normalization_35_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1936_conv2d_readvariableop_resource/
+conv2d_1936_biasadd_readvariableop_resource4
0batch_normalization_1936_readvariableop_resource6
2batch_normalization_1936_readvariableop_1_resourceE
Abatch_normalization_1936_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1968_conv2d_readvariableop_resource/
+conv2d_1968_biasadd_readvariableop_resource4
0batch_normalization_1968_readvariableop_resource6
2batch_normalization_1968_readvariableop_1_resourceE
Abatch_normalization_1968_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1968_fusedbatchnormv3_readvariableop_1_resourceG
Cconv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resourceH
Dconv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resourceM
Iconv_adjust_channels_235_batch_normalization_2155_readvariableop_resourceO
Kconv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resource^
Zconv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resource`
\conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_163_conv2d_readvariableop_resource.
*conv2d_163_biasadd_readvariableop_resource3
/batch_normalization_163_readvariableop_resource5
1batch_normalization_163_readvariableop_1_resourceD
@batch_normalization_163_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_163_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_195_conv2d_readvariableop_resource.
*conv2d_195_biasadd_readvariableop_resource3
/batch_normalization_195_readvariableop_resource5
1batch_normalization_195_readvariableop_1_resourceD
@batch_normalization_195_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_195_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1522_conv2d_readvariableop_resource/
+conv2d_1522_biasadd_readvariableop_resource4
0batch_normalization_1522_readvariableop_resource6
2batch_normalization_1522_readvariableop_1_resourceE
Abatch_normalization_1522_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1556_conv2d_readvariableop_resource/
+conv2d_1556_biasadd_readvariableop_resource4
0batch_normalization_1556_readvariableop_resource6
2batch_normalization_1556_readvariableop_1_resourceE
Abatch_normalization_1556_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource,
(dense_630_matmul_readvariableop_resource-
)dense_630_biasadd_readvariableop_resource,
(dense_631_matmul_readvariableop_resource-
)dense_631_biasadd_readvariableop_resource
identity??8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1522/ReadVariableOp?)batch_normalization_1522/ReadVariableOp_1?8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1556/ReadVariableOp?)batch_normalization_1556/ReadVariableOp_1?7batch_normalization_163/FusedBatchNormV3/ReadVariableOp?9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_163/ReadVariableOp?(batch_normalization_163/ReadVariableOp_1?8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1936/ReadVariableOp?)batch_normalization_1936/ReadVariableOp_1?7batch_normalization_195/FusedBatchNormV3/ReadVariableOp?9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_195/ReadVariableOp?(batch_normalization_195/ReadVariableOp_1?8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1968/ReadVariableOp?)batch_normalization_1968/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?"conv2d_1522/BiasAdd/ReadVariableOp?!conv2d_1522/Conv2D/ReadVariableOp?"conv2d_1556/BiasAdd/ReadVariableOp?!conv2d_1556/Conv2D/ReadVariableOp?!conv2d_163/BiasAdd/ReadVariableOp? conv2d_163/Conv2D/ReadVariableOp?"conv2d_1936/BiasAdd/ReadVariableOp?!conv2d_1936/Conv2D/ReadVariableOp?!conv2d_195/BiasAdd/ReadVariableOp? conv2d_195/Conv2D/ReadVariableOp?"conv2d_1968/BiasAdd/ReadVariableOp?!conv2d_1968/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp? dense_630/BiasAdd/ReadVariableOp?dense_630/MatMul/ReadVariableOp? dense_631/BiasAdd/ReadVariableOp?dense_631/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_6/Relu?
max_pooling2d_6/MaxPoolMaxPoolre_lu_6/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_6/MaxPool?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
re_lu_27/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_27/Relu?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2Dre_lu_27/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_35/BiasAdd?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_35/ReadVariableOp?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_35/ReadVariableOp_1?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_35/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_35/FusedBatchNormV3?
re_lu_35/ReluRelu+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_35/Relu?
!conv2d_1936/Conv2D/ReadVariableOpReadVariableOp*conv2d_1936_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_1936/Conv2D/ReadVariableOp?
conv2d_1936/Conv2DConv2Dre_lu_35/Relu:activations:0)conv2d_1936/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1936/Conv2D?
"conv2d_1936/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1936_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1936/BiasAdd/ReadVariableOp?
conv2d_1936/BiasAddBiasAddconv2d_1936/Conv2D:output:0*conv2d_1936/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1936/BiasAdd?
'batch_normalization_1936/ReadVariableOpReadVariableOp0batch_normalization_1936_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1936/ReadVariableOp?
)batch_normalization_1936/ReadVariableOp_1ReadVariableOp2batch_normalization_1936_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1936/ReadVariableOp_1?
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1936_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1936/FusedBatchNormV3FusedBatchNormV3conv2d_1936/BiasAdd:output:0/batch_normalization_1936/ReadVariableOp:value:01batch_normalization_1936/ReadVariableOp_1:value:0@batch_normalization_1936/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1936/FusedBatchNormV3?
re_lu_1763/ReluRelu-batch_normalization_1936/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1763/Relu?
!conv2d_1968/Conv2D/ReadVariableOpReadVariableOp*conv2d_1968_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1968/Conv2D/ReadVariableOp?
conv2d_1968/Conv2DConv2Dre_lu_1763/Relu:activations:0)conv2d_1968/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1968/Conv2D?
"conv2d_1968/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1968_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1968/BiasAdd/ReadVariableOp?
conv2d_1968/BiasAddBiasAddconv2d_1968/Conv2D:output:0*conv2d_1968/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1968/BiasAdd?
'batch_normalization_1968/ReadVariableOpReadVariableOp0batch_normalization_1968_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1968/ReadVariableOp?
)batch_normalization_1968/ReadVariableOp_1ReadVariableOp2batch_normalization_1968_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1968/ReadVariableOp_1?
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1968_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1968_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1968/FusedBatchNormV3FusedBatchNormV3conv2d_1968/BiasAdd:output:0/batch_normalization_1968/ReadVariableOp:value:01batch_normalization_1968/ReadVariableOp_1:value:0@batch_normalization_1968/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2+
)batch_normalization_1968/FusedBatchNormV3?
re_lu_1795/ReluRelu-batch_normalization_1968/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1795/Relu?
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOpReadVariableOpCconv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02<
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp?
+conv_adjust_channels_235/conv2d_2155/Conv2DConv2Dre_lu_1795/Relu:activations:0Bconv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2-
+conv_adjust_channels_235/conv2d_2155/Conv2D?
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOpReadVariableOpDconv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?
,conv_adjust_channels_235/conv2d_2155/BiasAddBiasAdd4conv_adjust_channels_235/conv2d_2155/Conv2D:output:0Cconv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2.
,conv_adjust_channels_235/conv2d_2155/BiasAdd?
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpReadVariableOpIconv_adjust_channels_235_batch_normalization_2155_readvariableop_resource*
_output_shapes
:*
dtype02B
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1ReadVariableOpKconv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02S
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02U
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?
Bconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3FusedBatchNormV35conv_adjust_channels_235/conv2d_2155/BiasAdd:output:0Hconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp:value:0Jconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1:value:0Yconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp:value:0[conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2D
Bconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3?
add_356/addAddV2re_lu_35/Relu:activations:0Fconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
add_356/add?
 conv2d_163/Conv2D/ReadVariableOpReadVariableOp)conv2d_163_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_163/Conv2D/ReadVariableOp?
conv2d_163/Conv2DConv2Dadd_356/add:z:0(conv2d_163/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_163/Conv2D?
!conv2d_163/BiasAdd/ReadVariableOpReadVariableOp*conv2d_163_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_163/BiasAdd/ReadVariableOp?
conv2d_163/BiasAddBiasAddconv2d_163/Conv2D:output:0)conv2d_163/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_163/BiasAdd?
&batch_normalization_163/ReadVariableOpReadVariableOp/batch_normalization_163_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_163/ReadVariableOp?
(batch_normalization_163/ReadVariableOp_1ReadVariableOp1batch_normalization_163_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_163/ReadVariableOp_1?
7batch_normalization_163/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_163_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_163/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_163_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_163/FusedBatchNormV3FusedBatchNormV3conv2d_163/BiasAdd:output:0.batch_normalization_163/ReadVariableOp:value:00batch_normalization_163/ReadVariableOp_1:value:0?batch_normalization_163/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_163/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2*
(batch_normalization_163/FusedBatchNormV3?
re_lu_163/ReluRelu,batch_normalization_163/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_163/Relu?
 conv2d_195/Conv2D/ReadVariableOpReadVariableOp)conv2d_195_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02"
 conv2d_195/Conv2D/ReadVariableOp?
conv2d_195/Conv2DConv2Dre_lu_163/Relu:activations:0(conv2d_195/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_195/Conv2D?
!conv2d_195/BiasAdd/ReadVariableOpReadVariableOp*conv2d_195_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_195/BiasAdd/ReadVariableOp?
conv2d_195/BiasAddBiasAddconv2d_195/Conv2D:output:0)conv2d_195/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_195/BiasAdd?
&batch_normalization_195/ReadVariableOpReadVariableOp/batch_normalization_195_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_195/ReadVariableOp?
(batch_normalization_195/ReadVariableOp_1ReadVariableOp1batch_normalization_195_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_195/ReadVariableOp_1?
7batch_normalization_195/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_195_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_195/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_195_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_195/FusedBatchNormV3FusedBatchNormV3conv2d_195/BiasAdd:output:0.batch_normalization_195/ReadVariableOp:value:00batch_normalization_195/ReadVariableOp_1:value:0?batch_normalization_195/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_195/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 2*
(batch_normalization_195/FusedBatchNormV3?
re_lu_195/ReluRelu,batch_normalization_195/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_195/Relu?
add_357/addAddV2re_lu_195/Relu:activations:0re_lu_1795/Relu:activations:0*
T0*/
_output_shapes
:?????????   2
add_357/add?
!conv2d_1522/Conv2D/ReadVariableOpReadVariableOp*conv2d_1522_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv2d_1522/Conv2D/ReadVariableOp?
conv2d_1522/Conv2DConv2Dadd_357/add:z:0)conv2d_1522/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_1522/Conv2D?
"conv2d_1522/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1522_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2d_1522/BiasAdd/ReadVariableOp?
conv2d_1522/BiasAddBiasAddconv2d_1522/Conv2D:output:0*conv2d_1522/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_1522/BiasAdd?
'batch_normalization_1522/ReadVariableOpReadVariableOp0batch_normalization_1522_readvariableop_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_1522/ReadVariableOp?
)batch_normalization_1522/ReadVariableOp_1ReadVariableOp2batch_normalization_1522_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_1522/ReadVariableOp_1?
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1522_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1522/FusedBatchNormV3FusedBatchNormV3conv2d_1522/BiasAdd:output:0/batch_normalization_1522/ReadVariableOp:value:01batch_normalization_1522/ReadVariableOp_1:value:0@batch_normalization_1522/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2+
)batch_normalization_1522/FusedBatchNormV3?
re_lu_1404/ReluRelu-batch_normalization_1522/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
re_lu_1404/Relu?
max_pooling2d_1404/MaxPoolMaxPoolre_lu_1404/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1404/MaxPool?
!conv2d_1556/Conv2D/ReadVariableOpReadVariableOp*conv2d_1556_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02#
!conv2d_1556/Conv2D/ReadVariableOp?
conv2d_1556/Conv2DConv2D#max_pooling2d_1404/MaxPool:output:0)conv2d_1556/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_1556/Conv2D?
"conv2d_1556/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1556_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2d_1556/BiasAdd/ReadVariableOp?
conv2d_1556/BiasAddBiasAddconv2d_1556/Conv2D:output:0*conv2d_1556/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_1556/BiasAdd?
'batch_normalization_1556/ReadVariableOpReadVariableOp0batch_normalization_1556_readvariableop_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_1556/ReadVariableOp?
)batch_normalization_1556/ReadVariableOp_1ReadVariableOp2batch_normalization_1556_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_1556/ReadVariableOp_1?
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1556_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1556/FusedBatchNormV3FusedBatchNormV3conv2d_1556/BiasAdd:output:0/batch_normalization_1556/ReadVariableOp:value:01batch_normalization_1556/ReadVariableOp_1:value:0@batch_normalization_1556/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2+
)batch_normalization_1556/FusedBatchNormV3?
re_lu_1438/ReluRelu-batch_normalization_1556/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
re_lu_1438/Relu?
max_pooling2d_1438/MaxPoolMaxPoolre_lu_1438/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1438/MaxPoolw
flatten_315/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_315/Const?
flatten_315/ReshapeReshape#max_pooling2d_1438/MaxPool:output:0flatten_315/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_315/Reshape?
dense_630/MatMul/ReadVariableOpReadVariableOp(dense_630_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02!
dense_630/MatMul/ReadVariableOp?
dense_630/MatMulMatMulflatten_315/Reshape:output:0'dense_630/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_630/MatMul?
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_630/BiasAdd/ReadVariableOp?
dense_630/BiasAddBiasAdddense_630/MatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_630/BiasAddw
dense_630/ReluReludense_630/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_630/Relu?
dense_631/MatMul/ReadVariableOpReadVariableOp(dense_631_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_631/MatMul/ReadVariableOp?
dense_631/MatMulMatMuldense_630/Relu:activations:0'dense_631/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_631/MatMul?
 dense_631/BiasAdd/ReadVariableOpReadVariableOp)dense_631_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_631/BiasAdd/ReadVariableOp?
dense_631/BiasAddBiasAdddense_631/MatMul:product:0(dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_631/BiasAdd
dense_631/SoftmaxSoftmaxdense_631/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_631/Softmax?
IdentityIdentitydense_631/Softmax:softmax:09^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1522/ReadVariableOp*^batch_normalization_1522/ReadVariableOp_19^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1556/ReadVariableOp*^batch_normalization_1556/ReadVariableOp_18^batch_normalization_163/FusedBatchNormV3/ReadVariableOp:^batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_163/ReadVariableOp)^batch_normalization_163/ReadVariableOp_19^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1936/ReadVariableOp*^batch_normalization_1936/ReadVariableOp_18^batch_normalization_195/FusedBatchNormV3/ReadVariableOp:^batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_195/ReadVariableOp)^batch_normalization_195/ReadVariableOp_19^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1968/ReadVariableOp*^batch_normalization_1968/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1#^conv2d_1522/BiasAdd/ReadVariableOp"^conv2d_1522/Conv2D/ReadVariableOp#^conv2d_1556/BiasAdd/ReadVariableOp"^conv2d_1556/Conv2D/ReadVariableOp"^conv2d_163/BiasAdd/ReadVariableOp!^conv2d_163/Conv2D/ReadVariableOp#^conv2d_1936/BiasAdd/ReadVariableOp"^conv2d_1936/Conv2D/ReadVariableOp"^conv2d_195/BiasAdd/ReadVariableOp!^conv2d_195/Conv2D/ReadVariableOp#^conv2d_1968/BiasAdd/ReadVariableOp"^conv2d_1968/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOpR^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpT^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1A^conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpC^conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1<^conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp;^conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp ^dense_630/MatMul/ReadVariableOp!^dense_631/BiasAdd/ReadVariableOp ^dense_631/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2t
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1522/ReadVariableOp'batch_normalization_1522/ReadVariableOp2V
)batch_normalization_1522/ReadVariableOp_1)batch_normalization_1522/ReadVariableOp_12t
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1556/ReadVariableOp'batch_normalization_1556/ReadVariableOp2V
)batch_normalization_1556/ReadVariableOp_1)batch_normalization_1556/ReadVariableOp_12r
7batch_normalization_163/FusedBatchNormV3/ReadVariableOp7batch_normalization_163/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_19batch_normalization_163/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_163/ReadVariableOp&batch_normalization_163/ReadVariableOp2T
(batch_normalization_163/ReadVariableOp_1(batch_normalization_163/ReadVariableOp_12t
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1936/ReadVariableOp'batch_normalization_1936/ReadVariableOp2V
)batch_normalization_1936/ReadVariableOp_1)batch_normalization_1936/ReadVariableOp_12r
7batch_normalization_195/FusedBatchNormV3/ReadVariableOp7batch_normalization_195/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_19batch_normalization_195/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_195/ReadVariableOp&batch_normalization_195/ReadVariableOp2T
(batch_normalization_195/ReadVariableOp_1(batch_normalization_195/ReadVariableOp_12t
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1968/ReadVariableOp'batch_normalization_1968/ReadVariableOp2V
)batch_normalization_1968/ReadVariableOp_1)batch_normalization_1968/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12H
"conv2d_1522/BiasAdd/ReadVariableOp"conv2d_1522/BiasAdd/ReadVariableOp2F
!conv2d_1522/Conv2D/ReadVariableOp!conv2d_1522/Conv2D/ReadVariableOp2H
"conv2d_1556/BiasAdd/ReadVariableOp"conv2d_1556/BiasAdd/ReadVariableOp2F
!conv2d_1556/Conv2D/ReadVariableOp!conv2d_1556/Conv2D/ReadVariableOp2F
!conv2d_163/BiasAdd/ReadVariableOp!conv2d_163/BiasAdd/ReadVariableOp2D
 conv2d_163/Conv2D/ReadVariableOp conv2d_163/Conv2D/ReadVariableOp2H
"conv2d_1936/BiasAdd/ReadVariableOp"conv2d_1936/BiasAdd/ReadVariableOp2F
!conv2d_1936/Conv2D/ReadVariableOp!conv2d_1936/Conv2D/ReadVariableOp2F
!conv2d_195/BiasAdd/ReadVariableOp!conv2d_195/BiasAdd/ReadVariableOp2D
 conv2d_195/Conv2D/ReadVariableOp conv2d_195/Conv2D/ReadVariableOp2H
"conv2d_1968/BiasAdd/ReadVariableOp"conv2d_1968/BiasAdd/ReadVariableOp2F
!conv2d_1968/Conv2D/ReadVariableOp!conv2d_1968/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2?
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpQconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp2?
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_12?
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp2?
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_12z
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp2x
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2B
dense_630/MatMul/ReadVariableOpdense_630/MatMul/ReadVariableOp2D
 dense_631/BiasAdd/ReadVariableOp dense_631/BiasAdd/ReadVariableOp2B
dense_631/MatMul/ReadVariableOpdense_631/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_195_layer_call_and_return_conditional_losses_7474405

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
?
a
E__inference_re_lu_35_layer_call_and_return_conditional_losses_7476949

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
?
c
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_7474739

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
??
??
"__inference__wrapped_model_7472387
input_15
1model_314_conv2d_6_conv2d_readvariableop_resource6
2model_314_conv2d_6_biasadd_readvariableop_resource;
7model_314_batch_normalization_6_readvariableop_resource=
9model_314_batch_normalization_6_readvariableop_1_resourceL
Hmodel_314_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceN
Jmodel_314_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource6
2model_314_conv2d_27_conv2d_readvariableop_resource7
3model_314_conv2d_27_biasadd_readvariableop_resource<
8model_314_batch_normalization_27_readvariableop_resource>
:model_314_batch_normalization_27_readvariableop_1_resourceM
Imodel_314_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceO
Kmodel_314_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource6
2model_314_conv2d_35_conv2d_readvariableop_resource7
3model_314_conv2d_35_biasadd_readvariableop_resource<
8model_314_batch_normalization_35_readvariableop_resource>
:model_314_batch_normalization_35_readvariableop_1_resourceM
Imodel_314_batch_normalization_35_fusedbatchnormv3_readvariableop_resourceO
Kmodel_314_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8
4model_314_conv2d_1936_conv2d_readvariableop_resource9
5model_314_conv2d_1936_biasadd_readvariableop_resource>
:model_314_batch_normalization_1936_readvariableop_resource@
<model_314_batch_normalization_1936_readvariableop_1_resourceO
Kmodel_314_batch_normalization_1936_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_314_batch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource8
4model_314_conv2d_1968_conv2d_readvariableop_resource9
5model_314_conv2d_1968_biasadd_readvariableop_resource>
:model_314_batch_normalization_1968_readvariableop_resource@
<model_314_batch_normalization_1968_readvariableop_1_resourceO
Kmodel_314_batch_normalization_1968_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_314_batch_normalization_1968_fusedbatchnormv3_readvariableop_1_resourceQ
Mmodel_314_conv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resourceR
Nmodel_314_conv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resourceW
Smodel_314_conv_adjust_channels_235_batch_normalization_2155_readvariableop_resourceY
Umodel_314_conv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resourceh
dmodel_314_conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resourcej
fmodel_314_conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource7
3model_314_conv2d_163_conv2d_readvariableop_resource8
4model_314_conv2d_163_biasadd_readvariableop_resource=
9model_314_batch_normalization_163_readvariableop_resource?
;model_314_batch_normalization_163_readvariableop_1_resourceN
Jmodel_314_batch_normalization_163_fusedbatchnormv3_readvariableop_resourceP
Lmodel_314_batch_normalization_163_fusedbatchnormv3_readvariableop_1_resource7
3model_314_conv2d_195_conv2d_readvariableop_resource8
4model_314_conv2d_195_biasadd_readvariableop_resource=
9model_314_batch_normalization_195_readvariableop_resource?
;model_314_batch_normalization_195_readvariableop_1_resourceN
Jmodel_314_batch_normalization_195_fusedbatchnormv3_readvariableop_resourceP
Lmodel_314_batch_normalization_195_fusedbatchnormv3_readvariableop_1_resource8
4model_314_conv2d_1522_conv2d_readvariableop_resource9
5model_314_conv2d_1522_biasadd_readvariableop_resource>
:model_314_batch_normalization_1522_readvariableop_resource@
<model_314_batch_normalization_1522_readvariableop_1_resourceO
Kmodel_314_batch_normalization_1522_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_314_batch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource8
4model_314_conv2d_1556_conv2d_readvariableop_resource9
5model_314_conv2d_1556_biasadd_readvariableop_resource>
:model_314_batch_normalization_1556_readvariableop_resource@
<model_314_batch_normalization_1556_readvariableop_1_resourceO
Kmodel_314_batch_normalization_1556_fusedbatchnormv3_readvariableop_resourceQ
Mmodel_314_batch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource6
2model_314_dense_630_matmul_readvariableop_resource7
3model_314_dense_630_biasadd_readvariableop_resource6
2model_314_dense_631_matmul_readvariableop_resource7
3model_314_dense_631_biasadd_readvariableop_resource
identity??Bmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?Dmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?1model_314/batch_normalization_1522/ReadVariableOp?3model_314/batch_normalization_1522/ReadVariableOp_1?Bmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?Dmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?1model_314/batch_normalization_1556/ReadVariableOp?3model_314/batch_normalization_1556/ReadVariableOp_1?Amodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp?Cmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?0model_314/batch_normalization_163/ReadVariableOp?2model_314/batch_normalization_163/ReadVariableOp_1?Bmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?Dmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?1model_314/batch_normalization_1936/ReadVariableOp?3model_314/batch_normalization_1936/ReadVariableOp_1?Amodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp?Cmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?0model_314/batch_normalization_195/ReadVariableOp?2model_314/batch_normalization_195/ReadVariableOp_1?Bmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?Dmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?1model_314/batch_normalization_1968/ReadVariableOp?3model_314/batch_normalization_1968/ReadVariableOp_1?@model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Bmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?/model_314/batch_normalization_27/ReadVariableOp?1model_314/batch_normalization_27/ReadVariableOp_1?@model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Bmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?/model_314/batch_normalization_35/ReadVariableOp?1model_314/batch_normalization_35/ReadVariableOp_1??model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Amodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?.model_314/batch_normalization_6/ReadVariableOp?0model_314/batch_normalization_6/ReadVariableOp_1?,model_314/conv2d_1522/BiasAdd/ReadVariableOp?+model_314/conv2d_1522/Conv2D/ReadVariableOp?,model_314/conv2d_1556/BiasAdd/ReadVariableOp?+model_314/conv2d_1556/Conv2D/ReadVariableOp?+model_314/conv2d_163/BiasAdd/ReadVariableOp?*model_314/conv2d_163/Conv2D/ReadVariableOp?,model_314/conv2d_1936/BiasAdd/ReadVariableOp?+model_314/conv2d_1936/Conv2D/ReadVariableOp?+model_314/conv2d_195/BiasAdd/ReadVariableOp?*model_314/conv2d_195/Conv2D/ReadVariableOp?,model_314/conv2d_1968/BiasAdd/ReadVariableOp?+model_314/conv2d_1968/Conv2D/ReadVariableOp?*model_314/conv2d_27/BiasAdd/ReadVariableOp?)model_314/conv2d_27/Conv2D/ReadVariableOp?*model_314/conv2d_35/BiasAdd/ReadVariableOp?)model_314/conv2d_35/Conv2D/ReadVariableOp?)model_314/conv2d_6/BiasAdd/ReadVariableOp?(model_314/conv2d_6/Conv2D/ReadVariableOp?[model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?]model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?Jmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?Emodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?Dmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp?*model_314/dense_630/BiasAdd/ReadVariableOp?)model_314/dense_630/MatMul/ReadVariableOp?*model_314/dense_631/BiasAdd/ReadVariableOp?)model_314/dense_631/MatMul/ReadVariableOp?
(model_314/conv2d_6/Conv2D/ReadVariableOpReadVariableOp1model_314_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_314/conv2d_6/Conv2D/ReadVariableOp?
model_314/conv2d_6/Conv2DConv2Dinput_10model_314/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_314/conv2d_6/Conv2D?
)model_314/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2model_314_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_314/conv2d_6/BiasAdd/ReadVariableOp?
model_314/conv2d_6/BiasAddBiasAdd"model_314/conv2d_6/Conv2D:output:01model_314/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_314/conv2d_6/BiasAdd?
.model_314/batch_normalization_6/ReadVariableOpReadVariableOp7model_314_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype020
.model_314/batch_normalization_6/ReadVariableOp?
0model_314/batch_normalization_6/ReadVariableOp_1ReadVariableOp9model_314_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_314/batch_normalization_6/ReadVariableOp_1?
?model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_314_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Amodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_314_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
0model_314/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3#model_314/conv2d_6/BiasAdd:output:06model_314/batch_normalization_6/ReadVariableOp:value:08model_314/batch_normalization_6/ReadVariableOp_1:value:0Gmodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Imodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 22
0model_314/batch_normalization_6/FusedBatchNormV3?
model_314/re_lu_6/ReluRelu4model_314/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_314/re_lu_6/Relu?
!model_314/max_pooling2d_6/MaxPoolMaxPool$model_314/re_lu_6/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2#
!model_314/max_pooling2d_6/MaxPool?
)model_314/conv2d_27/Conv2D/ReadVariableOpReadVariableOp2model_314_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)model_314/conv2d_27/Conv2D/ReadVariableOp?
model_314/conv2d_27/Conv2DConv2D*model_314/max_pooling2d_6/MaxPool:output:01model_314/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_314/conv2d_27/Conv2D?
*model_314/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp3model_314_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_314/conv2d_27/BiasAdd/ReadVariableOp?
model_314/conv2d_27/BiasAddBiasAdd#model_314/conv2d_27/Conv2D:output:02model_314/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_314/conv2d_27/BiasAdd?
/model_314/batch_normalization_27/ReadVariableOpReadVariableOp8model_314_batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype021
/model_314/batch_normalization_27/ReadVariableOp?
1model_314/batch_normalization_27/ReadVariableOp_1ReadVariableOp:model_314_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_314/batch_normalization_27/ReadVariableOp_1?
@model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_314_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Bmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_314_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
1model_314/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$model_314/conv2d_27/BiasAdd:output:07model_314/batch_normalization_27/ReadVariableOp:value:09model_314/batch_normalization_27/ReadVariableOp_1:value:0Hmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 23
1model_314/batch_normalization_27/FusedBatchNormV3?
model_314/re_lu_27/ReluRelu5model_314/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_314/re_lu_27/Relu?
)model_314/conv2d_35/Conv2D/ReadVariableOpReadVariableOp2model_314_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)model_314/conv2d_35/Conv2D/ReadVariableOp?
model_314/conv2d_35/Conv2DConv2D%model_314/re_lu_27/Relu:activations:01model_314/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
model_314/conv2d_35/Conv2D?
*model_314/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp3model_314_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_314/conv2d_35/BiasAdd/ReadVariableOp?
model_314/conv2d_35/BiasAddBiasAdd#model_314/conv2d_35/Conv2D:output:02model_314/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
model_314/conv2d_35/BiasAdd?
/model_314/batch_normalization_35/ReadVariableOpReadVariableOp8model_314_batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype021
/model_314/batch_normalization_35/ReadVariableOp?
1model_314/batch_normalization_35/ReadVariableOp_1ReadVariableOp:model_314_batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype023
1model_314/batch_normalization_35/ReadVariableOp_1?
@model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_314_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
Bmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_314_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
1model_314/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3$model_314/conv2d_35/BiasAdd:output:07model_314/batch_normalization_35/ReadVariableOp:value:09model_314/batch_normalization_35/ReadVariableOp_1:value:0Hmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Jmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 23
1model_314/batch_normalization_35/FusedBatchNormV3?
model_314/re_lu_35/ReluRelu5model_314/batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_314/re_lu_35/Relu?
+model_314/conv2d_1936/Conv2D/ReadVariableOpReadVariableOp4model_314_conv2d_1936_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+model_314/conv2d_1936/Conv2D/ReadVariableOp?
model_314/conv2d_1936/Conv2DConv2D%model_314/re_lu_35/Relu:activations:03model_314/conv2d_1936/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_314/conv2d_1936/Conv2D?
,model_314/conv2d_1936/BiasAdd/ReadVariableOpReadVariableOp5model_314_conv2d_1936_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_314/conv2d_1936/BiasAdd/ReadVariableOp?
model_314/conv2d_1936/BiasAddBiasAdd%model_314/conv2d_1936/Conv2D:output:04model_314/conv2d_1936/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_314/conv2d_1936/BiasAdd?
1model_314/batch_normalization_1936/ReadVariableOpReadVariableOp:model_314_batch_normalization_1936_readvariableop_resource*
_output_shapes
: *
dtype023
1model_314/batch_normalization_1936/ReadVariableOp?
3model_314/batch_normalization_1936/ReadVariableOp_1ReadVariableOp<model_314_batch_normalization_1936_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_314/batch_normalization_1936/ReadVariableOp_1?
Bmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_314_batch_normalization_1936_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?
Dmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_314_batch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?
3model_314/batch_normalization_1936/FusedBatchNormV3FusedBatchNormV3&model_314/conv2d_1936/BiasAdd:output:09model_314/batch_normalization_1936/ReadVariableOp:value:0;model_314/batch_normalization_1936/ReadVariableOp_1:value:0Jmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_314/batch_normalization_1936/FusedBatchNormV3?
model_314/re_lu_1763/ReluRelu7model_314/batch_normalization_1936/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_314/re_lu_1763/Relu?
+model_314/conv2d_1968/Conv2D/ReadVariableOpReadVariableOp4model_314_conv2d_1968_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+model_314/conv2d_1968/Conv2D/ReadVariableOp?
model_314/conv2d_1968/Conv2DConv2D'model_314/re_lu_1763/Relu:activations:03model_314/conv2d_1968/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_314/conv2d_1968/Conv2D?
,model_314/conv2d_1968/BiasAdd/ReadVariableOpReadVariableOp5model_314_conv2d_1968_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_314/conv2d_1968/BiasAdd/ReadVariableOp?
model_314/conv2d_1968/BiasAddBiasAdd%model_314/conv2d_1968/Conv2D:output:04model_314/conv2d_1968/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_314/conv2d_1968/BiasAdd?
1model_314/batch_normalization_1968/ReadVariableOpReadVariableOp:model_314_batch_normalization_1968_readvariableop_resource*
_output_shapes
: *
dtype023
1model_314/batch_normalization_1968/ReadVariableOp?
3model_314/batch_normalization_1968/ReadVariableOp_1ReadVariableOp<model_314_batch_normalization_1968_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model_314/batch_normalization_1968/ReadVariableOp_1?
Bmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_314_batch_normalization_1968_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?
Dmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_314_batch_normalization_1968_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?
3model_314/batch_normalization_1968/FusedBatchNormV3FusedBatchNormV3&model_314/conv2d_1968/BiasAdd:output:09model_314/batch_normalization_1968/ReadVariableOp:value:0;model_314/batch_normalization_1968/ReadVariableOp_1:value:0Jmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 25
3model_314/batch_normalization_1968/FusedBatchNormV3?
model_314/re_lu_1795/ReluRelu7model_314/batch_normalization_1968/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_314/re_lu_1795/Relu?
Dmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOpReadVariableOpMmodel_314_conv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02F
Dmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp?
5model_314/conv_adjust_channels_235/conv2d_2155/Conv2DConv2D'model_314/re_lu_1795/Relu:activations:0Lmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
27
5model_314/conv_adjust_channels_235/conv2d_2155/Conv2D?
Emodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOpReadVariableOpNmodel_314_conv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Emodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?
6model_314/conv_adjust_channels_235/conv2d_2155/BiasAddBiasAdd>model_314/conv_adjust_channels_235/conv2d_2155/Conv2D:output:0Mmodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  28
6model_314/conv_adjust_channels_235/conv2d_2155/BiasAdd?
Jmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpReadVariableOpSmodel_314_conv_adjust_channels_235_batch_normalization_2155_readvariableop_resource*
_output_shapes
:*
dtype02L
Jmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?
Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1ReadVariableOpUmodel_314_conv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?
[model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpReadVariableOpdmodel_314_conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02]
[model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?
]model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpfmodel_314_conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02_
]model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?
Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3FusedBatchNormV3?model_314/conv_adjust_channels_235/conv2d_2155/BiasAdd:output:0Rmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp:value:0Tmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1:value:0cmodel_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp:value:0emodel_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2N
Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3?
model_314/add_356/addAddV2%model_314/re_lu_35/Relu:activations:0Pmodel_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
model_314/add_356/add?
*model_314/conv2d_163/Conv2D/ReadVariableOpReadVariableOp3model_314_conv2d_163_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*model_314/conv2d_163/Conv2D/ReadVariableOp?
model_314/conv2d_163/Conv2DConv2Dmodel_314/add_356/add:z:02model_314/conv2d_163/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_314/conv2d_163/Conv2D?
+model_314/conv2d_163/BiasAdd/ReadVariableOpReadVariableOp4model_314_conv2d_163_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_314/conv2d_163/BiasAdd/ReadVariableOp?
model_314/conv2d_163/BiasAddBiasAdd$model_314/conv2d_163/Conv2D:output:03model_314/conv2d_163/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_314/conv2d_163/BiasAdd?
0model_314/batch_normalization_163/ReadVariableOpReadVariableOp9model_314_batch_normalization_163_readvariableop_resource*
_output_shapes
: *
dtype022
0model_314/batch_normalization_163/ReadVariableOp?
2model_314/batch_normalization_163/ReadVariableOp_1ReadVariableOp;model_314_batch_normalization_163_readvariableop_1_resource*
_output_shapes
: *
dtype024
2model_314/batch_normalization_163/ReadVariableOp_1?
Amodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOpReadVariableOpJmodel_314_batch_normalization_163_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Amodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp?
Cmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmodel_314_batch_normalization_163_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?
2model_314/batch_normalization_163/FusedBatchNormV3FusedBatchNormV3%model_314/conv2d_163/BiasAdd:output:08model_314/batch_normalization_163/ReadVariableOp:value:0:model_314/batch_normalization_163/ReadVariableOp_1:value:0Imodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp:value:0Kmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 24
2model_314/batch_normalization_163/FusedBatchNormV3?
model_314/re_lu_163/ReluRelu6model_314/batch_normalization_163/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_314/re_lu_163/Relu?
*model_314/conv2d_195/Conv2D/ReadVariableOpReadVariableOp3model_314_conv2d_195_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*model_314/conv2d_195/Conv2D/ReadVariableOp?
model_314/conv2d_195/Conv2DConv2D&model_314/re_lu_163/Relu:activations:02model_314/conv2d_195/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_314/conv2d_195/Conv2D?
+model_314/conv2d_195/BiasAdd/ReadVariableOpReadVariableOp4model_314_conv2d_195_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_314/conv2d_195/BiasAdd/ReadVariableOp?
model_314/conv2d_195/BiasAddBiasAdd$model_314/conv2d_195/Conv2D:output:03model_314/conv2d_195/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_314/conv2d_195/BiasAdd?
0model_314/batch_normalization_195/ReadVariableOpReadVariableOp9model_314_batch_normalization_195_readvariableop_resource*
_output_shapes
: *
dtype022
0model_314/batch_normalization_195/ReadVariableOp?
2model_314/batch_normalization_195/ReadVariableOp_1ReadVariableOp;model_314_batch_normalization_195_readvariableop_1_resource*
_output_shapes
: *
dtype024
2model_314/batch_normalization_195/ReadVariableOp_1?
Amodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOpReadVariableOpJmodel_314_batch_normalization_195_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Amodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp?
Cmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmodel_314_batch_normalization_195_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?
2model_314/batch_normalization_195/FusedBatchNormV3FusedBatchNormV3%model_314/conv2d_195/BiasAdd:output:08model_314/batch_normalization_195/ReadVariableOp:value:0:model_314/batch_normalization_195/ReadVariableOp_1:value:0Imodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp:value:0Kmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
is_training( 24
2model_314/batch_normalization_195/FusedBatchNormV3?
model_314/re_lu_195/ReluRelu6model_314/batch_normalization_195/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
model_314/re_lu_195/Relu?
model_314/add_357/addAddV2&model_314/re_lu_195/Relu:activations:0'model_314/re_lu_1795/Relu:activations:0*
T0*/
_output_shapes
:?????????   2
model_314/add_357/add?
+model_314/conv2d_1522/Conv2D/ReadVariableOpReadVariableOp4model_314_conv2d_1522_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+model_314/conv2d_1522/Conv2D/ReadVariableOp?
model_314/conv2d_1522/Conv2DConv2Dmodel_314/add_357/add:z:03model_314/conv2d_1522/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
model_314/conv2d_1522/Conv2D?
,model_314/conv2d_1522/BiasAdd/ReadVariableOpReadVariableOp5model_314_conv2d_1522_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_314/conv2d_1522/BiasAdd/ReadVariableOp?
model_314/conv2d_1522/BiasAddBiasAdd%model_314/conv2d_1522/Conv2D:output:04model_314/conv2d_1522/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
model_314/conv2d_1522/BiasAdd?
1model_314/batch_normalization_1522/ReadVariableOpReadVariableOp:model_314_batch_normalization_1522_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_314/batch_normalization_1522/ReadVariableOp?
3model_314/batch_normalization_1522/ReadVariableOp_1ReadVariableOp<model_314_batch_normalization_1522_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model_314/batch_normalization_1522/ReadVariableOp_1?
Bmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_314_batch_normalization_1522_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?
Dmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_314_batch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?
3model_314/batch_normalization_1522/FusedBatchNormV3FusedBatchNormV3&model_314/conv2d_1522/BiasAdd:output:09model_314/batch_normalization_1522/ReadVariableOp:value:0;model_314/batch_normalization_1522/ReadVariableOp_1:value:0Jmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 25
3model_314/batch_normalization_1522/FusedBatchNormV3?
model_314/re_lu_1404/ReluRelu7model_314/batch_normalization_1522/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
model_314/re_lu_1404/Relu?
$model_314/max_pooling2d_1404/MaxPoolMaxPool'model_314/re_lu_1404/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2&
$model_314/max_pooling2d_1404/MaxPool?
+model_314/conv2d_1556/Conv2D/ReadVariableOpReadVariableOp4model_314_conv2d_1556_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+model_314/conv2d_1556/Conv2D/ReadVariableOp?
model_314/conv2d_1556/Conv2DConv2D-model_314/max_pooling2d_1404/MaxPool:output:03model_314/conv2d_1556/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
model_314/conv2d_1556/Conv2D?
,model_314/conv2d_1556/BiasAdd/ReadVariableOpReadVariableOp5model_314_conv2d_1556_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_314/conv2d_1556/BiasAdd/ReadVariableOp?
model_314/conv2d_1556/BiasAddBiasAdd%model_314/conv2d_1556/Conv2D:output:04model_314/conv2d_1556/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
model_314/conv2d_1556/BiasAdd?
1model_314/batch_normalization_1556/ReadVariableOpReadVariableOp:model_314_batch_normalization_1556_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_314/batch_normalization_1556/ReadVariableOp?
3model_314/batch_normalization_1556/ReadVariableOp_1ReadVariableOp<model_314_batch_normalization_1556_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model_314/batch_normalization_1556/ReadVariableOp_1?
Bmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_314_batch_normalization_1556_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?
Dmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_314_batch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?
3model_314/batch_normalization_1556/FusedBatchNormV3FusedBatchNormV3&model_314/conv2d_1556/BiasAdd:output:09model_314/batch_normalization_1556/ReadVariableOp:value:0;model_314/batch_normalization_1556/ReadVariableOp_1:value:0Jmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp:value:0Lmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 25
3model_314/batch_normalization_1556/FusedBatchNormV3?
model_314/re_lu_1438/ReluRelu7model_314/batch_normalization_1556/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
model_314/re_lu_1438/Relu?
$model_314/max_pooling2d_1438/MaxPoolMaxPool'model_314/re_lu_1438/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2&
$model_314/max_pooling2d_1438/MaxPool?
model_314/flatten_315/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_314/flatten_315/Const?
model_314/flatten_315/ReshapeReshape-model_314/max_pooling2d_1438/MaxPool:output:0$model_314/flatten_315/Const:output:0*
T0*)
_output_shapes
:???????????2
model_314/flatten_315/Reshape?
)model_314/dense_630/MatMul/ReadVariableOpReadVariableOp2model_314_dense_630_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02+
)model_314/dense_630/MatMul/ReadVariableOp?
model_314/dense_630/MatMulMatMul&model_314/flatten_315/Reshape:output:01model_314/dense_630/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_314/dense_630/MatMul?
*model_314/dense_630/BiasAdd/ReadVariableOpReadVariableOp3model_314_dense_630_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model_314/dense_630/BiasAdd/ReadVariableOp?
model_314/dense_630/BiasAddBiasAdd$model_314/dense_630/MatMul:product:02model_314/dense_630/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_314/dense_630/BiasAdd?
model_314/dense_630/ReluRelu$model_314/dense_630/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_314/dense_630/Relu?
)model_314/dense_631/MatMul/ReadVariableOpReadVariableOp2model_314_dense_631_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02+
)model_314/dense_631/MatMul/ReadVariableOp?
model_314/dense_631/MatMulMatMul&model_314/dense_630/Relu:activations:01model_314/dense_631/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_314/dense_631/MatMul?
*model_314/dense_631/BiasAdd/ReadVariableOpReadVariableOp3model_314_dense_631_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*model_314/dense_631/BiasAdd/ReadVariableOp?
model_314/dense_631/BiasAddBiasAdd$model_314/dense_631/MatMul:product:02model_314/dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_314/dense_631/BiasAdd?
model_314/dense_631/SoftmaxSoftmax$model_314/dense_631/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model_314/dense_631/Softmax?
IdentityIdentity%model_314/dense_631/Softmax:softmax:0C^model_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOpE^model_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_12^model_314/batch_normalization_1522/ReadVariableOp4^model_314/batch_normalization_1522/ReadVariableOp_1C^model_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOpE^model_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_12^model_314/batch_normalization_1556/ReadVariableOp4^model_314/batch_normalization_1556/ReadVariableOp_1B^model_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOpD^model_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_11^model_314/batch_normalization_163/ReadVariableOp3^model_314/batch_normalization_163/ReadVariableOp_1C^model_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOpE^model_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_12^model_314/batch_normalization_1936/ReadVariableOp4^model_314/batch_normalization_1936/ReadVariableOp_1B^model_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOpD^model_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_11^model_314/batch_normalization_195/ReadVariableOp3^model_314/batch_normalization_195/ReadVariableOp_1C^model_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOpE^model_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_12^model_314/batch_normalization_1968/ReadVariableOp4^model_314/batch_normalization_1968/ReadVariableOp_1A^model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOpC^model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_10^model_314/batch_normalization_27/ReadVariableOp2^model_314/batch_normalization_27/ReadVariableOp_1A^model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOpC^model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_10^model_314/batch_normalization_35/ReadVariableOp2^model_314/batch_normalization_35/ReadVariableOp_1@^model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOpB^model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/^model_314/batch_normalization_6/ReadVariableOp1^model_314/batch_normalization_6/ReadVariableOp_1-^model_314/conv2d_1522/BiasAdd/ReadVariableOp,^model_314/conv2d_1522/Conv2D/ReadVariableOp-^model_314/conv2d_1556/BiasAdd/ReadVariableOp,^model_314/conv2d_1556/Conv2D/ReadVariableOp,^model_314/conv2d_163/BiasAdd/ReadVariableOp+^model_314/conv2d_163/Conv2D/ReadVariableOp-^model_314/conv2d_1936/BiasAdd/ReadVariableOp,^model_314/conv2d_1936/Conv2D/ReadVariableOp,^model_314/conv2d_195/BiasAdd/ReadVariableOp+^model_314/conv2d_195/Conv2D/ReadVariableOp-^model_314/conv2d_1968/BiasAdd/ReadVariableOp,^model_314/conv2d_1968/Conv2D/ReadVariableOp+^model_314/conv2d_27/BiasAdd/ReadVariableOp*^model_314/conv2d_27/Conv2D/ReadVariableOp+^model_314/conv2d_35/BiasAdd/ReadVariableOp*^model_314/conv2d_35/Conv2D/ReadVariableOp*^model_314/conv2d_6/BiasAdd/ReadVariableOp)^model_314/conv2d_6/Conv2D/ReadVariableOp\^model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp^^model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1K^model_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpM^model_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1F^model_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOpE^model_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp+^model_314/dense_630/BiasAdd/ReadVariableOp*^model_314/dense_630/MatMul/ReadVariableOp+^model_314/dense_631/BiasAdd/ReadVariableOp*^model_314/dense_631/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2?
Bmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOpBmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp2?
Dmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1Dmodel_314/batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_12f
1model_314/batch_normalization_1522/ReadVariableOp1model_314/batch_normalization_1522/ReadVariableOp2j
3model_314/batch_normalization_1522/ReadVariableOp_13model_314/batch_normalization_1522/ReadVariableOp_12?
Bmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOpBmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp2?
Dmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1Dmodel_314/batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_12f
1model_314/batch_normalization_1556/ReadVariableOp1model_314/batch_normalization_1556/ReadVariableOp2j
3model_314/batch_normalization_1556/ReadVariableOp_13model_314/batch_normalization_1556/ReadVariableOp_12?
Amodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOpAmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp2?
Cmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1Cmodel_314/batch_normalization_163/FusedBatchNormV3/ReadVariableOp_12d
0model_314/batch_normalization_163/ReadVariableOp0model_314/batch_normalization_163/ReadVariableOp2h
2model_314/batch_normalization_163/ReadVariableOp_12model_314/batch_normalization_163/ReadVariableOp_12?
Bmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOpBmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp2?
Dmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1Dmodel_314/batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_12f
1model_314/batch_normalization_1936/ReadVariableOp1model_314/batch_normalization_1936/ReadVariableOp2j
3model_314/batch_normalization_1936/ReadVariableOp_13model_314/batch_normalization_1936/ReadVariableOp_12?
Amodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOpAmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp2?
Cmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1Cmodel_314/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_12d
0model_314/batch_normalization_195/ReadVariableOp0model_314/batch_normalization_195/ReadVariableOp2h
2model_314/batch_normalization_195/ReadVariableOp_12model_314/batch_normalization_195/ReadVariableOp_12?
Bmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOpBmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp2?
Dmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1Dmodel_314/batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_12f
1model_314/batch_normalization_1968/ReadVariableOp1model_314/batch_normalization_1968/ReadVariableOp2j
3model_314/batch_normalization_1968/ReadVariableOp_13model_314/batch_normalization_1968/ReadVariableOp_12?
@model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp@model_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Bmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Bmodel_314/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12b
/model_314/batch_normalization_27/ReadVariableOp/model_314/batch_normalization_27/ReadVariableOp2f
1model_314/batch_normalization_27/ReadVariableOp_11model_314/batch_normalization_27/ReadVariableOp_12?
@model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp@model_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Bmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Bmodel_314/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12b
/model_314/batch_normalization_35/ReadVariableOp/model_314/batch_normalization_35/ReadVariableOp2f
1model_314/batch_normalization_35/ReadVariableOp_11model_314/batch_normalization_35/ReadVariableOp_12?
?model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?model_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Amodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Amodel_314/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12`
.model_314/batch_normalization_6/ReadVariableOp.model_314/batch_normalization_6/ReadVariableOp2d
0model_314/batch_normalization_6/ReadVariableOp_10model_314/batch_normalization_6/ReadVariableOp_12\
,model_314/conv2d_1522/BiasAdd/ReadVariableOp,model_314/conv2d_1522/BiasAdd/ReadVariableOp2Z
+model_314/conv2d_1522/Conv2D/ReadVariableOp+model_314/conv2d_1522/Conv2D/ReadVariableOp2\
,model_314/conv2d_1556/BiasAdd/ReadVariableOp,model_314/conv2d_1556/BiasAdd/ReadVariableOp2Z
+model_314/conv2d_1556/Conv2D/ReadVariableOp+model_314/conv2d_1556/Conv2D/ReadVariableOp2Z
+model_314/conv2d_163/BiasAdd/ReadVariableOp+model_314/conv2d_163/BiasAdd/ReadVariableOp2X
*model_314/conv2d_163/Conv2D/ReadVariableOp*model_314/conv2d_163/Conv2D/ReadVariableOp2\
,model_314/conv2d_1936/BiasAdd/ReadVariableOp,model_314/conv2d_1936/BiasAdd/ReadVariableOp2Z
+model_314/conv2d_1936/Conv2D/ReadVariableOp+model_314/conv2d_1936/Conv2D/ReadVariableOp2Z
+model_314/conv2d_195/BiasAdd/ReadVariableOp+model_314/conv2d_195/BiasAdd/ReadVariableOp2X
*model_314/conv2d_195/Conv2D/ReadVariableOp*model_314/conv2d_195/Conv2D/ReadVariableOp2\
,model_314/conv2d_1968/BiasAdd/ReadVariableOp,model_314/conv2d_1968/BiasAdd/ReadVariableOp2Z
+model_314/conv2d_1968/Conv2D/ReadVariableOp+model_314/conv2d_1968/Conv2D/ReadVariableOp2X
*model_314/conv2d_27/BiasAdd/ReadVariableOp*model_314/conv2d_27/BiasAdd/ReadVariableOp2V
)model_314/conv2d_27/Conv2D/ReadVariableOp)model_314/conv2d_27/Conv2D/ReadVariableOp2X
*model_314/conv2d_35/BiasAdd/ReadVariableOp*model_314/conv2d_35/BiasAdd/ReadVariableOp2V
)model_314/conv2d_35/Conv2D/ReadVariableOp)model_314/conv2d_35/Conv2D/ReadVariableOp2V
)model_314/conv2d_6/BiasAdd/ReadVariableOp)model_314/conv2d_6/BiasAdd/ReadVariableOp2T
(model_314/conv2d_6/Conv2D/ReadVariableOp(model_314/conv2d_6/Conv2D/ReadVariableOp2?
[model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp[model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp2?
]model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1]model_314/conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_12?
Jmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpJmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp2?
Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1Lmodel_314/conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_12?
Emodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOpEmodel_314/conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp2?
Dmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOpDmodel_314/conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp2X
*model_314/dense_630/BiasAdd/ReadVariableOp*model_314/dense_630/BiasAdd/ReadVariableOp2V
)model_314/dense_630/MatMul/ReadVariableOp)model_314/dense_630/MatMul/ReadVariableOp2X
*model_314/dense_631/BiasAdd/ReadVariableOp*model_314/dense_631/BiasAdd/ReadVariableOp2V
)model_314/dense_631/MatMul/ReadVariableOp)model_314/dense_631/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7474440

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476697

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
:__inference_batch_normalization_1556_layer_call_fn_7477917

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
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_74736022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
n
D__inference_add_357_layer_call_and_return_conditional_losses_7474513

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
׷
?
F__inference_model_314_layer_call_and_return_conditional_losses_7475156

inputs
conv2d_6_7474991
conv2d_6_7474993!
batch_normalization_6_7474996!
batch_normalization_6_7474998!
batch_normalization_6_7475000!
batch_normalization_6_7475002
conv2d_27_7475007
conv2d_27_7475009"
batch_normalization_27_7475012"
batch_normalization_27_7475014"
batch_normalization_27_7475016"
batch_normalization_27_7475018
conv2d_35_7475022
conv2d_35_7475024"
batch_normalization_35_7475027"
batch_normalization_35_7475029"
batch_normalization_35_7475031"
batch_normalization_35_7475033
conv2d_1936_7475037
conv2d_1936_7475039$
 batch_normalization_1936_7475042$
 batch_normalization_1936_7475044$
 batch_normalization_1936_7475046$
 batch_normalization_1936_7475048
conv2d_1968_7475052
conv2d_1968_7475054$
 batch_normalization_1968_7475057$
 batch_normalization_1968_7475059$
 batch_normalization_1968_7475061$
 batch_normalization_1968_7475063$
 conv_adjust_channels_235_7475067$
 conv_adjust_channels_235_7475069$
 conv_adjust_channels_235_7475071$
 conv_adjust_channels_235_7475073$
 conv_adjust_channels_235_7475075$
 conv_adjust_channels_235_7475077
conv2d_163_7475081
conv2d_163_7475083#
batch_normalization_163_7475086#
batch_normalization_163_7475088#
batch_normalization_163_7475090#
batch_normalization_163_7475092
conv2d_195_7475096
conv2d_195_7475098#
batch_normalization_195_7475101#
batch_normalization_195_7475103#
batch_normalization_195_7475105#
batch_normalization_195_7475107
conv2d_1522_7475112
conv2d_1522_7475114$
 batch_normalization_1522_7475117$
 batch_normalization_1522_7475119$
 batch_normalization_1522_7475121$
 batch_normalization_1522_7475123
conv2d_1556_7475128
conv2d_1556_7475130$
 batch_normalization_1556_7475133$
 batch_normalization_1556_7475135$
 batch_normalization_1556_7475137$
 batch_normalization_1556_7475139
dense_630_7475145
dense_630_7475147
dense_631_7475150
dense_631_7475152
identity??0batch_normalization_1522/StatefulPartitionedCall?0batch_normalization_1556/StatefulPartitionedCall?/batch_normalization_163/StatefulPartitionedCall?0batch_normalization_1936/StatefulPartitionedCall?/batch_normalization_195/StatefulPartitionedCall?0batch_normalization_1968/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?#conv2d_1522/StatefulPartitionedCall?#conv2d_1556/StatefulPartitionedCall?"conv2d_163/StatefulPartitionedCall?#conv2d_1936/StatefulPartitionedCall?"conv2d_195/StatefulPartitionedCall?#conv2d_1968/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?0conv_adjust_channels_235/StatefulPartitionedCall?!dense_630/StatefulPartitionedCall?!dense_631/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_7474991conv2d_6_7474993*
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_74736702"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_7474996batch_normalization_6_7474998batch_normalization_6_7475000batch_normalization_6_7475002*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737052/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_74737642
re_lu_6/PartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_74724972!
max_pooling2d_6/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_27_7475007conv2d_27_7475009*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_74737832#
!conv2d_27/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_27_7475012batch_normalization_27_7475014batch_normalization_27_7475016batch_normalization_27_7475018*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_747381820
.batch_normalization_27/StatefulPartitionedCall?
re_lu_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_74738772
re_lu_27/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall!re_lu_27/PartitionedCall:output:0conv2d_35_7475022conv2d_35_7475024*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_74738952#
!conv2d_35/StatefulPartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_7475027batch_normalization_35_7475029batch_normalization_35_7475031batch_normalization_35_7475033*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_747393020
.batch_normalization_35/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_74739892
re_lu_35/PartitionedCall?
#conv2d_1936/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0conv2d_1936_7475037conv2d_1936_7475039*
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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_74740072%
#conv2d_1936/StatefulPartitionedCall?
0batch_normalization_1936/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1936/StatefulPartitionedCall:output:0 batch_normalization_1936_7475042 batch_normalization_1936_7475044 batch_normalization_1936_7475046 batch_normalization_1936_7475048*
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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_747404222
0batch_normalization_1936/StatefulPartitionedCall?
re_lu_1763/PartitionedCallPartitionedCall9batch_normalization_1936/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_74741012
re_lu_1763/PartitionedCall?
#conv2d_1968/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1763/PartitionedCall:output:0conv2d_1968_7475052conv2d_1968_7475054*
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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_74741192%
#conv2d_1968/StatefulPartitionedCall?
0batch_normalization_1968/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1968/StatefulPartitionedCall:output:0 batch_normalization_1968_7475057 batch_normalization_1968_7475059 batch_normalization_1968_7475061 batch_normalization_1968_7475063*
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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_747415422
0batch_normalization_1968/StatefulPartitionedCall?
re_lu_1795/PartitionedCallPartitionedCall9batch_normalization_1968/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_74742132
re_lu_1795/PartitionedCall?
0conv_adjust_channels_235/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1795/PartitionedCall:output:0 conv_adjust_channels_235_7475067 conv_adjust_channels_235_7475069 conv_adjust_channels_235_7475071 conv_adjust_channels_235_7475073 conv_adjust_channels_235_7475075 conv_adjust_channels_235_7475077*
Tin
	2*
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
GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_747316622
0conv_adjust_channels_235/StatefulPartitionedCall?
add_356/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:09conv_adjust_channels_235/StatefulPartitionedCall:output:0*
Tin
2*
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
D__inference_add_356_layer_call_and_return_conditional_losses_74742742
add_356/PartitionedCall?
"conv2d_163/StatefulPartitionedCallStatefulPartitionedCall add_356/PartitionedCall:output:0conv2d_163_7475081conv2d_163_7475083*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_163_layer_call_and_return_conditional_losses_74742932$
"conv2d_163/StatefulPartitionedCall?
/batch_normalization_163/StatefulPartitionedCallStatefulPartitionedCall+conv2d_163/StatefulPartitionedCall:output:0batch_normalization_163_7475086batch_normalization_163_7475088batch_normalization_163_7475090batch_normalization_163_7475092*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_747432821
/batch_normalization_163/StatefulPartitionedCall?
re_lu_163/PartitionedCallPartitionedCall8batch_normalization_163/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_163_layer_call_and_return_conditional_losses_74743872
re_lu_163/PartitionedCall?
"conv2d_195/StatefulPartitionedCallStatefulPartitionedCall"re_lu_163/PartitionedCall:output:0conv2d_195_7475096conv2d_195_7475098*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_195_layer_call_and_return_conditional_losses_74744052$
"conv2d_195/StatefulPartitionedCall?
/batch_normalization_195/StatefulPartitionedCallStatefulPartitionedCall+conv2d_195/StatefulPartitionedCall:output:0batch_normalization_195_7475101batch_normalization_195_7475103batch_normalization_195_7475105batch_normalization_195_7475107*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_747444021
/batch_normalization_195/StatefulPartitionedCall?
re_lu_195/PartitionedCallPartitionedCall8batch_normalization_195/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_195_layer_call_and_return_conditional_losses_74744992
re_lu_195/PartitionedCall?
add_357/PartitionedCallPartitionedCall"re_lu_195/PartitionedCall:output:0#re_lu_1795/PartitionedCall:output:0*
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
D__inference_add_357_layer_call_and_return_conditional_losses_74745132
add_357/PartitionedCall?
#conv2d_1522/StatefulPartitionedCallStatefulPartitionedCall add_357/PartitionedCall:output:0conv2d_1522_7475112conv2d_1522_7475114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_74745322%
#conv2d_1522/StatefulPartitionedCall?
0batch_normalization_1522/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1522/StatefulPartitionedCall:output:0 batch_normalization_1522_7475117 batch_normalization_1522_7475119 batch_normalization_1522_7475121 batch_normalization_1522_7475123*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_747456722
0batch_normalization_1522/StatefulPartitionedCall?
re_lu_1404/PartitionedCallPartitionedCall9batch_normalization_1522/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_74746262
re_lu_1404/PartitionedCall?
"max_pooling2d_1404/PartitionedCallPartitionedCall#re_lu_1404/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_74735342$
"max_pooling2d_1404/PartitionedCall?
#conv2d_1556/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1404/PartitionedCall:output:0conv2d_1556_7475128conv2d_1556_7475130*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_74746452%
#conv2d_1556/StatefulPartitionedCall?
0batch_normalization_1556/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1556/StatefulPartitionedCall:output:0 batch_normalization_1556_7475133 batch_normalization_1556_7475135 batch_normalization_1556_7475137 batch_normalization_1556_7475139*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_747468022
0batch_normalization_1556/StatefulPartitionedCall?
re_lu_1438/PartitionedCallPartitionedCall9batch_normalization_1556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_74747392
re_lu_1438/PartitionedCall?
"max_pooling2d_1438/PartitionedCallPartitionedCall#re_lu_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_74736502$
"max_pooling2d_1438/PartitionedCall?
flatten_315/PartitionedCallPartitionedCall+max_pooling2d_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_315_layer_call_and_return_conditional_losses_74747542
flatten_315/PartitionedCall?
!dense_630/StatefulPartitionedCallStatefulPartitionedCall$flatten_315/PartitionedCall:output:0dense_630_7475145dense_630_7475147*
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
F__inference_dense_630_layer_call_and_return_conditional_losses_74747732#
!dense_630/StatefulPartitionedCall?
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_7475150dense_631_7475152*
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
F__inference_dense_631_layer_call_and_return_conditional_losses_74748002#
!dense_631/StatefulPartitionedCall?
IdentityIdentity*dense_631/StatefulPartitionedCall:output:01^batch_normalization_1522/StatefulPartitionedCall1^batch_normalization_1556/StatefulPartitionedCall0^batch_normalization_163/StatefulPartitionedCall1^batch_normalization_1936/StatefulPartitionedCall0^batch_normalization_195/StatefulPartitionedCall1^batch_normalization_1968/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall$^conv2d_1522/StatefulPartitionedCall$^conv2d_1556/StatefulPartitionedCall#^conv2d_163/StatefulPartitionedCall$^conv2d_1936/StatefulPartitionedCall#^conv2d_195/StatefulPartitionedCall$^conv2d_1968/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall1^conv_adjust_channels_235/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2d
0batch_normalization_1522/StatefulPartitionedCall0batch_normalization_1522/StatefulPartitionedCall2d
0batch_normalization_1556/StatefulPartitionedCall0batch_normalization_1556/StatefulPartitionedCall2b
/batch_normalization_163/StatefulPartitionedCall/batch_normalization_163/StatefulPartitionedCall2d
0batch_normalization_1936/StatefulPartitionedCall0batch_normalization_1936/StatefulPartitionedCall2b
/batch_normalization_195/StatefulPartitionedCall/batch_normalization_195/StatefulPartitionedCall2d
0batch_normalization_1968/StatefulPartitionedCall0batch_normalization_1968/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2J
#conv2d_1522/StatefulPartitionedCall#conv2d_1522/StatefulPartitionedCall2J
#conv2d_1556/StatefulPartitionedCall#conv2d_1556/StatefulPartitionedCall2H
"conv2d_163/StatefulPartitionedCall"conv2d_163/StatefulPartitionedCall2J
#conv2d_1936/StatefulPartitionedCall#conv2d_1936/StatefulPartitionedCall2H
"conv2d_195/StatefulPartitionedCall"conv2d_195/StatefulPartitionedCall2J
#conv2d_1968/StatefulPartitionedCall#conv2d_1968/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2d
0conv_adjust_channels_235/StatefulPartitionedCall0conv_adjust_channels_235/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1556_layer_call_fn_7477994

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
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_74746982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7474328

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
F__inference_dense_630_layer_call_and_return_conditional_losses_7478026

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
n
D__inference_add_356_layer_call_and_return_conditional_losses_7474274

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????  2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????  :?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_195_layer_call_and_return_conditional_losses_7477531

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
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7473818

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
?

*__inference_conv2d_6_layer_call_fn_7476502

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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_74736702
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
?
?
:__inference_batch_normalization_1556_layer_call_fn_7477930

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
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_74736332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7473413

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
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7474567

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?)
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477294

inputs.
*conv2d_2155_conv2d_readvariableop_resource/
+conv2d_2155_biasadd_readvariableop_resource4
0batch_normalization_2155_readvariableop_resource6
2batch_normalization_2155_readvariableop_1_resourceE
Abatch_normalization_2155_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource
identity??'batch_normalization_2155/AssignNewValue?)batch_normalization_2155/AssignNewValue_1?8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_2155/ReadVariableOp?)batch_normalization_2155/ReadVariableOp_1?"conv2d_2155/BiasAdd/ReadVariableOp?!conv2d_2155/Conv2D/ReadVariableOp?
!conv2d_2155/Conv2D/ReadVariableOpReadVariableOp*conv2d_2155_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_2155/Conv2D/ReadVariableOp?
conv2d_2155/Conv2DConv2Dinputs)conv2d_2155/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_2155/Conv2D?
"conv2d_2155/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"conv2d_2155/BiasAdd/ReadVariableOp?
conv2d_2155/BiasAddBiasAddconv2d_2155/Conv2D:output:0*conv2d_2155/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_2155/BiasAdd?
'batch_normalization_2155/ReadVariableOpReadVariableOp0batch_normalization_2155_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization_2155/ReadVariableOp?
)batch_normalization_2155/ReadVariableOp_1ReadVariableOp2batch_normalization_2155_readvariableop_1_resource*
_output_shapes
:*
dtype02+
)batch_normalization_2155/ReadVariableOp_1?
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_2155_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_2155/FusedBatchNormV3FusedBatchNormV3conv2d_2155/BiasAdd:output:0/batch_normalization_2155/ReadVariableOp:value:01batch_normalization_2155/ReadVariableOp_1:value:0@batch_normalization_2155/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_2155/FusedBatchNormV3?
'batch_normalization_2155/AssignNewValueAssignVariableOpAbatch_normalization_2155_fusedbatchnormv3_readvariableop_resource6batch_normalization_2155/FusedBatchNormV3:batch_mean:09^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_2155/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_2155/AssignNewValue?
)batch_normalization_2155/AssignNewValue_1AssignVariableOpCbatch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_2155/FusedBatchNormV3:batch_variance:0;^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_2155/AssignNewValue_1?
IdentityIdentity-batch_normalization_2155/FusedBatchNormV3:y:0(^batch_normalization_2155/AssignNewValue*^batch_normalization_2155/AssignNewValue_19^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp;^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_2155/ReadVariableOp*^batch_normalization_2155/ReadVariableOp_1#^conv2d_2155/BiasAdd/ReadVariableOp"^conv2d_2155/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2R
'batch_normalization_2155/AssignNewValue'batch_normalization_2155/AssignNewValue2V
)batch_normalization_2155/AssignNewValue_1)batch_normalization_2155/AssignNewValue_12t
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_2155/ReadVariableOp'batch_normalization_2155/ReadVariableOp2V
)batch_normalization_2155/ReadVariableOp_1)batch_normalization_2155/ReadVariableOp_12H
"conv2d_2155/BiasAdd/ReadVariableOp"conv2d_2155/BiasAdd/ReadVariableOp2F
!conv2d_2155/Conv2D/ReadVariableOp!conv2d_2155/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7473602

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_7473534

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
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7473836

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
?	
?
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_7478065

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
b
F__inference_re_lu_195_layer_call_and_return_conditional_losses_7477673

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
??
?=
F__inference_model_314_layer_call_and_return_conditional_losses_7475983

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource2
.batch_normalization_35_readvariableop_resource4
0batch_normalization_35_readvariableop_1_resourceC
?batch_normalization_35_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1936_conv2d_readvariableop_resource/
+conv2d_1936_biasadd_readvariableop_resource4
0batch_normalization_1936_readvariableop_resource6
2batch_normalization_1936_readvariableop_1_resourceE
Abatch_normalization_1936_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1968_conv2d_readvariableop_resource/
+conv2d_1968_biasadd_readvariableop_resource4
0batch_normalization_1968_readvariableop_resource6
2batch_normalization_1968_readvariableop_1_resourceE
Abatch_normalization_1968_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1968_fusedbatchnormv3_readvariableop_1_resourceG
Cconv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resourceH
Dconv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resourceM
Iconv_adjust_channels_235_batch_normalization_2155_readvariableop_resourceO
Kconv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resource^
Zconv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resource`
\conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_163_conv2d_readvariableop_resource.
*conv2d_163_biasadd_readvariableop_resource3
/batch_normalization_163_readvariableop_resource5
1batch_normalization_163_readvariableop_1_resourceD
@batch_normalization_163_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_163_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_195_conv2d_readvariableop_resource.
*conv2d_195_biasadd_readvariableop_resource3
/batch_normalization_195_readvariableop_resource5
1batch_normalization_195_readvariableop_1_resourceD
@batch_normalization_195_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_195_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1522_conv2d_readvariableop_resource/
+conv2d_1522_biasadd_readvariableop_resource4
0batch_normalization_1522_readvariableop_resource6
2batch_normalization_1522_readvariableop_1_resourceE
Abatch_normalization_1522_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource.
*conv2d_1556_conv2d_readvariableop_resource/
+conv2d_1556_biasadd_readvariableop_resource4
0batch_normalization_1556_readvariableop_resource6
2batch_normalization_1556_readvariableop_1_resourceE
Abatch_normalization_1556_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource,
(dense_630_matmul_readvariableop_resource-
)dense_630_biasadd_readvariableop_resource,
(dense_631_matmul_readvariableop_resource-
)dense_631_biasadd_readvariableop_resource
identity??'batch_normalization_1522/AssignNewValue?)batch_normalization_1522/AssignNewValue_1?8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1522/ReadVariableOp?)batch_normalization_1522/ReadVariableOp_1?'batch_normalization_1556/AssignNewValue?)batch_normalization_1556/AssignNewValue_1?8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1556/ReadVariableOp?)batch_normalization_1556/ReadVariableOp_1?&batch_normalization_163/AssignNewValue?(batch_normalization_163/AssignNewValue_1?7batch_normalization_163/FusedBatchNormV3/ReadVariableOp?9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_163/ReadVariableOp?(batch_normalization_163/ReadVariableOp_1?'batch_normalization_1936/AssignNewValue?)batch_normalization_1936/AssignNewValue_1?8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1936/ReadVariableOp?)batch_normalization_1936/ReadVariableOp_1?&batch_normalization_195/AssignNewValue?(batch_normalization_195/AssignNewValue_1?7batch_normalization_195/FusedBatchNormV3/ReadVariableOp?9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_195/ReadVariableOp?(batch_normalization_195/ReadVariableOp_1?'batch_normalization_1968/AssignNewValue?)batch_normalization_1968/AssignNewValue_1?8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_1968/ReadVariableOp?)batch_normalization_1968/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_35/AssignNewValue?'batch_normalization_35/AssignNewValue_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?"conv2d_1522/BiasAdd/ReadVariableOp?!conv2d_1522/Conv2D/ReadVariableOp?"conv2d_1556/BiasAdd/ReadVariableOp?!conv2d_1556/Conv2D/ReadVariableOp?!conv2d_163/BiasAdd/ReadVariableOp? conv2d_163/Conv2D/ReadVariableOp?"conv2d_1936/BiasAdd/ReadVariableOp?!conv2d_1936/Conv2D/ReadVariableOp?!conv2d_195/BiasAdd/ReadVariableOp? conv2d_195/Conv2D/ReadVariableOp?"conv2d_1968/BiasAdd/ReadVariableOp?!conv2d_1968/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?@conv_adjust_channels_235/batch_normalization_2155/AssignNewValue?Bconv_adjust_channels_235/batch_normalization_2155/AssignNewValue_1?Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp? dense_630/BiasAdd/ReadVariableOp?dense_630/MatMul/ReadVariableOp? dense_631/BiasAdd/ReadVariableOp?dense_631/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_6/Relu?
max_pooling2d_6/MaxPoolMaxPoolre_lu_6/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_6/MaxPool?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
re_lu_27/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_27/Relu?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2Dre_lu_27/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_35/BiasAdd?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_35/ReadVariableOp?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_35/ReadVariableOp_1?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_35/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_35/FusedBatchNormV3?
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_35/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_35/AssignNewValue?
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_35/AssignNewValue_1?
re_lu_35/ReluRelu+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
re_lu_35/Relu?
!conv2d_1936/Conv2D/ReadVariableOpReadVariableOp*conv2d_1936_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_1936/Conv2D/ReadVariableOp?
conv2d_1936/Conv2DConv2Dre_lu_35/Relu:activations:0)conv2d_1936/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1936/Conv2D?
"conv2d_1936/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1936_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1936/BiasAdd/ReadVariableOp?
conv2d_1936/BiasAddBiasAddconv2d_1936/Conv2D:output:0*conv2d_1936/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1936/BiasAdd?
'batch_normalization_1936/ReadVariableOpReadVariableOp0batch_normalization_1936_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1936/ReadVariableOp?
)batch_normalization_1936/ReadVariableOp_1ReadVariableOp2batch_normalization_1936_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1936/ReadVariableOp_1?
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1936_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1936/FusedBatchNormV3FusedBatchNormV3conv2d_1936/BiasAdd:output:0/batch_normalization_1936/ReadVariableOp:value:01batch_normalization_1936/ReadVariableOp_1:value:0@batch_normalization_1936/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1936/FusedBatchNormV3?
'batch_normalization_1936/AssignNewValueAssignVariableOpAbatch_normalization_1936_fusedbatchnormv3_readvariableop_resource6batch_normalization_1936/FusedBatchNormV3:batch_mean:09^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1936/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1936/AssignNewValue?
)batch_normalization_1936/AssignNewValue_1AssignVariableOpCbatch_normalization_1936_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1936/FusedBatchNormV3:batch_variance:0;^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1936/AssignNewValue_1?
re_lu_1763/ReluRelu-batch_normalization_1936/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1763/Relu?
!conv2d_1968/Conv2D/ReadVariableOpReadVariableOp*conv2d_1968_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!conv2d_1968/Conv2D/ReadVariableOp?
conv2d_1968/Conv2DConv2Dre_lu_1763/Relu:activations:0)conv2d_1968/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_1968/Conv2D?
"conv2d_1968/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1968_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv2d_1968/BiasAdd/ReadVariableOp?
conv2d_1968/BiasAddBiasAddconv2d_1968/Conv2D:output:0*conv2d_1968/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_1968/BiasAdd?
'batch_normalization_1968/ReadVariableOpReadVariableOp0batch_normalization_1968_readvariableop_resource*
_output_shapes
: *
dtype02)
'batch_normalization_1968/ReadVariableOp?
)batch_normalization_1968/ReadVariableOp_1ReadVariableOp2batch_normalization_1968_readvariableop_1_resource*
_output_shapes
: *
dtype02+
)batch_normalization_1968/ReadVariableOp_1?
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1968_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02:
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1968_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1968/FusedBatchNormV3FusedBatchNormV3conv2d_1968/BiasAdd:output:0/batch_normalization_1968/ReadVariableOp:value:01batch_normalization_1968/ReadVariableOp_1:value:0@batch_normalization_1968/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1968/FusedBatchNormV3?
'batch_normalization_1968/AssignNewValueAssignVariableOpAbatch_normalization_1968_fusedbatchnormv3_readvariableop_resource6batch_normalization_1968/FusedBatchNormV3:batch_mean:09^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1968/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1968/AssignNewValue?
)batch_normalization_1968/AssignNewValue_1AssignVariableOpCbatch_normalization_1968_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1968/FusedBatchNormV3:batch_variance:0;^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1968/AssignNewValue_1?
re_lu_1795/ReluRelu-batch_normalization_1968/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_1795/Relu?
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOpReadVariableOpCconv_adjust_channels_235_conv2d_2155_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02<
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp?
+conv_adjust_channels_235/conv2d_2155/Conv2DConv2Dre_lu_1795/Relu:activations:0Bconv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2-
+conv_adjust_channels_235/conv2d_2155/Conv2D?
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOpReadVariableOpDconv_adjust_channels_235_conv2d_2155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp?
,conv_adjust_channels_235/conv2d_2155/BiasAddBiasAdd4conv_adjust_channels_235/conv2d_2155/Conv2D:output:0Cconv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2.
,conv_adjust_channels_235/conv2d_2155/BiasAdd?
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpReadVariableOpIconv_adjust_channels_235_batch_normalization_2155_readvariableop_resource*
_output_shapes
:*
dtype02B
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp?
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1ReadVariableOpKconv_adjust_channels_235_batch_normalization_2155_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1?
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02S
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02U
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?
Bconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3FusedBatchNormV35conv_adjust_channels_235/conv2d_2155/BiasAdd:output:0Hconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp:value:0Jconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1:value:0Yconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp:value:0[conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2D
Bconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3?
@conv_adjust_channels_235/batch_normalization_2155/AssignNewValueAssignVariableOpZconv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_resourceOconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3:batch_mean:0R^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*m
_classc
a_loc:@conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02B
@conv_adjust_channels_235/batch_normalization_2155/AssignNewValue?
Bconv_adjust_channels_235/batch_normalization_2155/AssignNewValue_1AssignVariableOp\conv_adjust_channels_235_batch_normalization_2155_fusedbatchnormv3_readvariableop_1_resourceSconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3:batch_variance:0T^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*o
_classe
caloc:@conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02D
Bconv_adjust_channels_235/batch_normalization_2155/AssignNewValue_1?
add_356/addAddV2re_lu_35/Relu:activations:0Fconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
add_356/add?
 conv2d_163/Conv2D/ReadVariableOpReadVariableOp)conv2d_163_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_163/Conv2D/ReadVariableOp?
conv2d_163/Conv2DConv2Dadd_356/add:z:0(conv2d_163/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_163/Conv2D?
!conv2d_163/BiasAdd/ReadVariableOpReadVariableOp*conv2d_163_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_163/BiasAdd/ReadVariableOp?
conv2d_163/BiasAddBiasAddconv2d_163/Conv2D:output:0)conv2d_163/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_163/BiasAdd?
&batch_normalization_163/ReadVariableOpReadVariableOp/batch_normalization_163_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_163/ReadVariableOp?
(batch_normalization_163/ReadVariableOp_1ReadVariableOp1batch_normalization_163_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_163/ReadVariableOp_1?
7batch_normalization_163/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_163_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_163/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_163_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_163/FusedBatchNormV3FusedBatchNormV3conv2d_163/BiasAdd:output:0.batch_normalization_163/ReadVariableOp:value:00batch_normalization_163/ReadVariableOp_1:value:0?batch_normalization_163/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_163/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_163/FusedBatchNormV3?
&batch_normalization_163/AssignNewValueAssignVariableOp@batch_normalization_163_fusedbatchnormv3_readvariableop_resource5batch_normalization_163/FusedBatchNormV3:batch_mean:08^batch_normalization_163/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_163/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_163/AssignNewValue?
(batch_normalization_163/AssignNewValue_1AssignVariableOpBbatch_normalization_163_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_163/FusedBatchNormV3:batch_variance:0:^batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_163/AssignNewValue_1?
re_lu_163/ReluRelu,batch_normalization_163/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_163/Relu?
 conv2d_195/Conv2D/ReadVariableOpReadVariableOp)conv2d_195_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02"
 conv2d_195/Conv2D/ReadVariableOp?
conv2d_195/Conv2DConv2Dre_lu_163/Relu:activations:0(conv2d_195/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_195/Conv2D?
!conv2d_195/BiasAdd/ReadVariableOpReadVariableOp*conv2d_195_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_195/BiasAdd/ReadVariableOp?
conv2d_195/BiasAddBiasAddconv2d_195/Conv2D:output:0)conv2d_195/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_195/BiasAdd?
&batch_normalization_195/ReadVariableOpReadVariableOp/batch_normalization_195_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_195/ReadVariableOp?
(batch_normalization_195/ReadVariableOp_1ReadVariableOp1batch_normalization_195_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_195/ReadVariableOp_1?
7batch_normalization_195/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_195_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_195/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_195_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_195/FusedBatchNormV3FusedBatchNormV3conv2d_195/BiasAdd:output:0.batch_normalization_195/ReadVariableOp:value:00batch_normalization_195/ReadVariableOp_1:value:0?batch_normalization_195/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_195/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????   : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_195/FusedBatchNormV3?
&batch_normalization_195/AssignNewValueAssignVariableOp@batch_normalization_195_fusedbatchnormv3_readvariableop_resource5batch_normalization_195/FusedBatchNormV3:batch_mean:08^batch_normalization_195/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_195/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_195/AssignNewValue?
(batch_normalization_195/AssignNewValue_1AssignVariableOpBbatch_normalization_195_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_195/FusedBatchNormV3:batch_variance:0:^batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_195/AssignNewValue_1?
re_lu_195/ReluRelu,batch_normalization_195/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????   2
re_lu_195/Relu?
add_357/addAddV2re_lu_195/Relu:activations:0re_lu_1795/Relu:activations:0*
T0*/
_output_shapes
:?????????   2
add_357/add?
!conv2d_1522/Conv2D/ReadVariableOpReadVariableOp*conv2d_1522_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv2d_1522/Conv2D/ReadVariableOp?
conv2d_1522/Conv2DConv2Dadd_357/add:z:0)conv2d_1522/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_1522/Conv2D?
"conv2d_1522/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1522_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2d_1522/BiasAdd/ReadVariableOp?
conv2d_1522/BiasAddBiasAddconv2d_1522/Conv2D:output:0*conv2d_1522/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_1522/BiasAdd?
'batch_normalization_1522/ReadVariableOpReadVariableOp0batch_normalization_1522_readvariableop_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_1522/ReadVariableOp?
)batch_normalization_1522/ReadVariableOp_1ReadVariableOp2batch_normalization_1522_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_1522/ReadVariableOp_1?
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1522_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1522/FusedBatchNormV3FusedBatchNormV3conv2d_1522/BiasAdd:output:0/batch_normalization_1522/ReadVariableOp:value:01batch_normalization_1522/ReadVariableOp_1:value:0@batch_normalization_1522/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1522/FusedBatchNormV3?
'batch_normalization_1522/AssignNewValueAssignVariableOpAbatch_normalization_1522_fusedbatchnormv3_readvariableop_resource6batch_normalization_1522/FusedBatchNormV3:batch_mean:09^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1522/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1522/AssignNewValue?
)batch_normalization_1522/AssignNewValue_1AssignVariableOpCbatch_normalization_1522_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1522/FusedBatchNormV3:batch_variance:0;^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1522/AssignNewValue_1?
re_lu_1404/ReluRelu-batch_normalization_1522/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
re_lu_1404/Relu?
max_pooling2d_1404/MaxPoolMaxPoolre_lu_1404/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1404/MaxPool?
!conv2d_1556/Conv2D/ReadVariableOpReadVariableOp*conv2d_1556_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02#
!conv2d_1556/Conv2D/ReadVariableOp?
conv2d_1556/Conv2DConv2D#max_pooling2d_1404/MaxPool:output:0)conv2d_1556/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_1556/Conv2D?
"conv2d_1556/BiasAdd/ReadVariableOpReadVariableOp+conv2d_1556_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2d_1556/BiasAdd/ReadVariableOp?
conv2d_1556/BiasAddBiasAddconv2d_1556/Conv2D:output:0*conv2d_1556/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_1556/BiasAdd?
'batch_normalization_1556/ReadVariableOpReadVariableOp0batch_normalization_1556_readvariableop_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_1556/ReadVariableOp?
)batch_normalization_1556/ReadVariableOp_1ReadVariableOp2batch_normalization_1556_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_1556/ReadVariableOp_1?
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_1556_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_1556/FusedBatchNormV3FusedBatchNormV3conv2d_1556/BiasAdd:output:0/batch_normalization_1556/ReadVariableOp:value:01batch_normalization_1556/ReadVariableOp_1:value:0@batch_normalization_1556/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2+
)batch_normalization_1556/FusedBatchNormV3?
'batch_normalization_1556/AssignNewValueAssignVariableOpAbatch_normalization_1556_fusedbatchnormv3_readvariableop_resource6batch_normalization_1556/FusedBatchNormV3:batch_mean:09^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_1556/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'batch_normalization_1556/AssignNewValue?
)batch_normalization_1556/AssignNewValue_1AssignVariableOpCbatch_normalization_1556_fusedbatchnormv3_readvariableop_1_resource:batch_normalization_1556/FusedBatchNormV3:batch_variance:0;^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)batch_normalization_1556/AssignNewValue_1?
re_lu_1438/ReluRelu-batch_normalization_1556/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
re_lu_1438/Relu?
max_pooling2d_1438/MaxPoolMaxPoolre_lu_1438/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1438/MaxPoolw
flatten_315/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_315/Const?
flatten_315/ReshapeReshape#max_pooling2d_1438/MaxPool:output:0flatten_315/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_315/Reshape?
dense_630/MatMul/ReadVariableOpReadVariableOp(dense_630_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02!
dense_630/MatMul/ReadVariableOp?
dense_630/MatMulMatMulflatten_315/Reshape:output:0'dense_630/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_630/MatMul?
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_630/BiasAdd/ReadVariableOp?
dense_630/BiasAddBiasAdddense_630/MatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_630/BiasAddw
dense_630/ReluReludense_630/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_630/Relu?
dense_631/MatMul/ReadVariableOpReadVariableOp(dense_631_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_631/MatMul/ReadVariableOp?
dense_631/MatMulMatMuldense_630/Relu:activations:0'dense_631/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_631/MatMul?
 dense_631/BiasAdd/ReadVariableOpReadVariableOp)dense_631_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_631/BiasAdd/ReadVariableOp?
dense_631/BiasAddBiasAdddense_631/MatMul:product:0(dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_631/BiasAdd
dense_631/SoftmaxSoftmaxdense_631/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_631/Softmax?
IdentityIdentitydense_631/Softmax:softmax:0(^batch_normalization_1522/AssignNewValue*^batch_normalization_1522/AssignNewValue_19^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1522/ReadVariableOp*^batch_normalization_1522/ReadVariableOp_1(^batch_normalization_1556/AssignNewValue*^batch_normalization_1556/AssignNewValue_19^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1556/ReadVariableOp*^batch_normalization_1556/ReadVariableOp_1'^batch_normalization_163/AssignNewValue)^batch_normalization_163/AssignNewValue_18^batch_normalization_163/FusedBatchNormV3/ReadVariableOp:^batch_normalization_163/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_163/ReadVariableOp)^batch_normalization_163/ReadVariableOp_1(^batch_normalization_1936/AssignNewValue*^batch_normalization_1936/AssignNewValue_19^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1936/ReadVariableOp*^batch_normalization_1936/ReadVariableOp_1'^batch_normalization_195/AssignNewValue)^batch_normalization_195/AssignNewValue_18^batch_normalization_195/FusedBatchNormV3/ReadVariableOp:^batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_195/ReadVariableOp)^batch_normalization_195/ReadVariableOp_1(^batch_normalization_1968/AssignNewValue*^batch_normalization_1968/AssignNewValue_19^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp;^batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_1968/ReadVariableOp*^batch_normalization_1968/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1#^conv2d_1522/BiasAdd/ReadVariableOp"^conv2d_1522/Conv2D/ReadVariableOp#^conv2d_1556/BiasAdd/ReadVariableOp"^conv2d_1556/Conv2D/ReadVariableOp"^conv2d_163/BiasAdd/ReadVariableOp!^conv2d_163/Conv2D/ReadVariableOp#^conv2d_1936/BiasAdd/ReadVariableOp"^conv2d_1936/Conv2D/ReadVariableOp"^conv2d_195/BiasAdd/ReadVariableOp!^conv2d_195/Conv2D/ReadVariableOp#^conv2d_1968/BiasAdd/ReadVariableOp"^conv2d_1968/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOpA^conv_adjust_channels_235/batch_normalization_2155/AssignNewValueC^conv_adjust_channels_235/batch_normalization_2155/AssignNewValue_1R^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpT^conv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1A^conv_adjust_channels_235/batch_normalization_2155/ReadVariableOpC^conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1<^conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp;^conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp ^dense_630/MatMul/ReadVariableOp!^dense_631/BiasAdd/ReadVariableOp ^dense_631/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2R
'batch_normalization_1522/AssignNewValue'batch_normalization_1522/AssignNewValue2V
)batch_normalization_1522/AssignNewValue_1)batch_normalization_1522/AssignNewValue_12t
8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp8batch_normalization_1522/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1522/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1522/ReadVariableOp'batch_normalization_1522/ReadVariableOp2V
)batch_normalization_1522/ReadVariableOp_1)batch_normalization_1522/ReadVariableOp_12R
'batch_normalization_1556/AssignNewValue'batch_normalization_1556/AssignNewValue2V
)batch_normalization_1556/AssignNewValue_1)batch_normalization_1556/AssignNewValue_12t
8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp8batch_normalization_1556/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1556/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1556/ReadVariableOp'batch_normalization_1556/ReadVariableOp2V
)batch_normalization_1556/ReadVariableOp_1)batch_normalization_1556/ReadVariableOp_12P
&batch_normalization_163/AssignNewValue&batch_normalization_163/AssignNewValue2T
(batch_normalization_163/AssignNewValue_1(batch_normalization_163/AssignNewValue_12r
7batch_normalization_163/FusedBatchNormV3/ReadVariableOp7batch_normalization_163/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_163/FusedBatchNormV3/ReadVariableOp_19batch_normalization_163/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_163/ReadVariableOp&batch_normalization_163/ReadVariableOp2T
(batch_normalization_163/ReadVariableOp_1(batch_normalization_163/ReadVariableOp_12R
'batch_normalization_1936/AssignNewValue'batch_normalization_1936/AssignNewValue2V
)batch_normalization_1936/AssignNewValue_1)batch_normalization_1936/AssignNewValue_12t
8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp8batch_normalization_1936/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1936/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1936/ReadVariableOp'batch_normalization_1936/ReadVariableOp2V
)batch_normalization_1936/ReadVariableOp_1)batch_normalization_1936/ReadVariableOp_12P
&batch_normalization_195/AssignNewValue&batch_normalization_195/AssignNewValue2T
(batch_normalization_195/AssignNewValue_1(batch_normalization_195/AssignNewValue_12r
7batch_normalization_195/FusedBatchNormV3/ReadVariableOp7batch_normalization_195/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_195/FusedBatchNormV3/ReadVariableOp_19batch_normalization_195/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_195/ReadVariableOp&batch_normalization_195/ReadVariableOp2T
(batch_normalization_195/ReadVariableOp_1(batch_normalization_195/ReadVariableOp_12R
'batch_normalization_1968/AssignNewValue'batch_normalization_1968/AssignNewValue2V
)batch_normalization_1968/AssignNewValue_1)batch_normalization_1968/AssignNewValue_12t
8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp8batch_normalization_1968/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_1968/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_1968/ReadVariableOp'batch_normalization_1968/ReadVariableOp2V
)batch_normalization_1968/ReadVariableOp_1)batch_normalization_1968/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12H
"conv2d_1522/BiasAdd/ReadVariableOp"conv2d_1522/BiasAdd/ReadVariableOp2F
!conv2d_1522/Conv2D/ReadVariableOp!conv2d_1522/Conv2D/ReadVariableOp2H
"conv2d_1556/BiasAdd/ReadVariableOp"conv2d_1556/BiasAdd/ReadVariableOp2F
!conv2d_1556/Conv2D/ReadVariableOp!conv2d_1556/Conv2D/ReadVariableOp2F
!conv2d_163/BiasAdd/ReadVariableOp!conv2d_163/BiasAdd/ReadVariableOp2D
 conv2d_163/Conv2D/ReadVariableOp conv2d_163/Conv2D/ReadVariableOp2H
"conv2d_1936/BiasAdd/ReadVariableOp"conv2d_1936/BiasAdd/ReadVariableOp2F
!conv2d_1936/Conv2D/ReadVariableOp!conv2d_1936/Conv2D/ReadVariableOp2F
!conv2d_195/BiasAdd/ReadVariableOp!conv2d_195/BiasAdd/ReadVariableOp2D
 conv2d_195/Conv2D/ReadVariableOp conv2d_195/Conv2D/ReadVariableOp2H
"conv2d_1968/BiasAdd/ReadVariableOp"conv2d_1968/BiasAdd/ReadVariableOp2F
!conv2d_1968/Conv2D/ReadVariableOp!conv2d_1968/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2?
@conv_adjust_channels_235/batch_normalization_2155/AssignNewValue@conv_adjust_channels_235/batch_normalization_2155/AssignNewValue2?
Bconv_adjust_channels_235/batch_normalization_2155/AssignNewValue_1Bconv_adjust_channels_235/batch_normalization_2155/AssignNewValue_12?
Qconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOpQconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp2?
Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1Sconv_adjust_channels_235/batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_12?
@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp@conv_adjust_channels_235/batch_normalization_2155/ReadVariableOp2?
Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_1Bconv_adjust_channels_235/batch_normalization_2155/ReadVariableOp_12z
;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp;conv_adjust_channels_235/conv2d_2155/BiasAdd/ReadVariableOp2x
:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp:conv_adjust_channels_235/conv2d_2155/Conv2D/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2B
dense_630/MatMul/ReadVariableOpdense_630/MatMul/ReadVariableOp2D
 dense_631/BiasAdd/ReadVariableOp dense_631/BiasAdd/ReadVariableOp2B
dense_631/MatMul/ReadVariableOpdense_631/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
:__inference_conv_adjust_channels_235_layer_call_fn_7473181
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_74731662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????   
!
_user_specified_name	input_1
?
?
8__inference_batch_normalization_27_layer_call_fn_7476787

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_74725962
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
,__inference_re_lu_1763_layer_call_fn_7477111

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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_74741012
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
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_7474645

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7472596

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
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477467

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
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478176

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
:__inference_batch_normalization_1936_layer_call_fn_7477088

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_74727732
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
?
?
9__inference_batch_normalization_195_layer_call_fn_7477604

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_74744582
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
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7474680

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7472908

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
?
?
9__inference_batch_normalization_195_layer_call_fn_7477668

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_74734132
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
?
?
:__inference_batch_normalization_1522_layer_call_fn_7477837

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
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_74745852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477624

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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_7474119

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
?
d
H__inference_flatten_315_layer_call_and_return_conditional_losses_7474754

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_7476650

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
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476836

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
?
?
:__inference_conv_adjust_channels_235_layer_call_fn_7477352

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_74732012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
a
E__inference_re_lu_27_layer_call_and_return_conditional_losses_7473877

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
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477904

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1968_layer_call_fn_7477258

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_74741722
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
?
?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7473633

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7474458

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
:__inference_batch_normalization_1968_layer_call_fn_7477245

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_74741542
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
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476854

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
,__inference_conv2d_163_layer_call_fn_7477383

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
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_163_layer_call_and_return_conditional_losses_74742932
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
?	
?
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_7473037

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477318

inputs.
*conv2d_2155_conv2d_readvariableop_resource/
+conv2d_2155_biasadd_readvariableop_resource4
0batch_normalization_2155_readvariableop_resource6
2batch_normalization_2155_readvariableop_1_resourceE
Abatch_normalization_2155_fusedbatchnormv3_readvariableop_resourceG
Cbatch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource
identity??8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization_2155/ReadVariableOp?)batch_normalization_2155/ReadVariableOp_1?"conv2d_2155/BiasAdd/ReadVariableOp?!conv2d_2155/Conv2D/ReadVariableOp?
!conv2d_2155/Conv2D/ReadVariableOpReadVariableOp*conv2d_2155_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv2d_2155/Conv2D/ReadVariableOp?
conv2d_2155/Conv2DConv2Dinputs)conv2d_2155/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_2155/Conv2D?
"conv2d_2155/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"conv2d_2155/BiasAdd/ReadVariableOp?
conv2d_2155/BiasAddBiasAddconv2d_2155/Conv2D:output:0*conv2d_2155/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_2155/BiasAdd?
'batch_normalization_2155/ReadVariableOpReadVariableOp0batch_normalization_2155_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization_2155/ReadVariableOp?
)batch_normalization_2155/ReadVariableOp_1ReadVariableOp2batch_normalization_2155_readvariableop_1_resource*
_output_shapes
:*
dtype02+
)batch_normalization_2155/ReadVariableOp_1?
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_2155_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp?
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_2155_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1?
)batch_normalization_2155/FusedBatchNormV3FusedBatchNormV3conv2d_2155/BiasAdd:output:0/batch_normalization_2155/ReadVariableOp:value:01batch_normalization_2155/ReadVariableOp_1:value:0@batch_normalization_2155/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2+
)batch_normalization_2155/FusedBatchNormV3?
IdentityIdentity-batch_normalization_2155/FusedBatchNormV3:y:09^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp;^batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization_2155/ReadVariableOp*^batch_normalization_2155/ReadVariableOp_1#^conv2d_2155/BiasAdd/ReadVariableOp"^conv2d_2155/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2t
8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp8batch_normalization_2155/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_1:batch_normalization_2155/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_2155/ReadVariableOp'batch_normalization_2155/ReadVariableOp2V
)batch_normalization_2155/ReadVariableOp_1)batch_normalization_2155/ReadVariableOp_12H
"conv2d_2155/BiasAdd/ReadVariableOp"conv2d_2155/BiasAdd/ReadVariableOp2F
!conv2d_2155/Conv2D/ReadVariableOp!conv2d_2155/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1968_layer_call_fn_7477194

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_74729082
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
?
?
:__inference_batch_normalization_1556_layer_call_fn_7477981

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
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_74746802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477560

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
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7473012

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
?	
?
F__inference_conv2d_35_layer_call_and_return_conditional_losses_7473895

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
9__inference_batch_normalization_195_layer_call_fn_7477655

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_74733822
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
?
?
8__inference_batch_normalization_27_layer_call_fn_7476710

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_74738182
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
?
F
*__inference_re_lu_27_layer_call_fn_7476797

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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_74738772
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
?
?
8__inference_batch_normalization_35_layer_call_fn_7476944

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_74739482
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
?
?
:__inference_batch_normalization_1936_layer_call_fn_7477024

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_74740422
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
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7472804

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
?
c
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_7477842

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477057

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
G
+__inference_re_lu_163_layer_call_fn_7477521

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
GPU 2J 8? *O
fJRH
F__inference_re_lu_163_layer_call_and_return_conditional_losses_74743872
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
?
P
4__inference_max_pooling2d_1438_layer_call_fn_7473656

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
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_74736502
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
?
?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477968

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478112

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
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7474346

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
?
d
H__inference_flatten_315_layer_call_and_return_conditional_losses_7478010

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_7473783

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
?
c
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_7477106

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
,__inference_conv2d_195_layer_call_fn_7477540

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
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_195_layer_call_and_return_conditional_losses_74744052
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
?4
 __inference__traced_save_7478537
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop1
-savev2_conv2d_1936_kernel_read_readvariableop/
+savev2_conv2d_1936_bias_read_readvariableop=
9savev2_batch_normalization_1936_gamma_read_readvariableop<
8savev2_batch_normalization_1936_beta_read_readvariableopC
?savev2_batch_normalization_1936_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1936_moving_variance_read_readvariableop1
-savev2_conv2d_1968_kernel_read_readvariableop/
+savev2_conv2d_1968_bias_read_readvariableop=
9savev2_batch_normalization_1968_gamma_read_readvariableop<
8savev2_batch_normalization_1968_beta_read_readvariableopC
?savev2_batch_normalization_1968_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1968_moving_variance_read_readvariableop0
,savev2_conv2d_163_kernel_read_readvariableop.
*savev2_conv2d_163_bias_read_readvariableop<
8savev2_batch_normalization_163_gamma_read_readvariableop;
7savev2_batch_normalization_163_beta_read_readvariableopB
>savev2_batch_normalization_163_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_163_moving_variance_read_readvariableop0
,savev2_conv2d_195_kernel_read_readvariableop.
*savev2_conv2d_195_bias_read_readvariableop<
8savev2_batch_normalization_195_gamma_read_readvariableop;
7savev2_batch_normalization_195_beta_read_readvariableopB
>savev2_batch_normalization_195_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_195_moving_variance_read_readvariableop1
-savev2_conv2d_1522_kernel_read_readvariableop/
+savev2_conv2d_1522_bias_read_readvariableop=
9savev2_batch_normalization_1522_gamma_read_readvariableop<
8savev2_batch_normalization_1522_beta_read_readvariableopC
?savev2_batch_normalization_1522_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1522_moving_variance_read_readvariableop1
-savev2_conv2d_1556_kernel_read_readvariableop/
+savev2_conv2d_1556_bias_read_readvariableop=
9savev2_batch_normalization_1556_gamma_read_readvariableop<
8savev2_batch_normalization_1556_beta_read_readvariableopC
?savev2_batch_normalization_1556_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1556_moving_variance_read_readvariableop/
+savev2_dense_630_kernel_read_readvariableop-
)savev2_dense_630_bias_read_readvariableop/
+savev2_dense_631_kernel_read_readvariableop-
)savev2_dense_631_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableopJ
Fsavev2_conv_adjust_channels_235_conv2d_2155_kernel_read_readvariableopH
Dsavev2_conv_adjust_channels_235_conv2d_2155_bias_read_readvariableopV
Rsavev2_conv_adjust_channels_235_batch_normalization_2155_gamma_read_readvariableopU
Qsavev2_conv_adjust_channels_235_batch_normalization_2155_beta_read_readvariableop\
Xsavev2_conv_adjust_channels_235_batch_normalization_2155_moving_mean_read_readvariableop`
\savev2_conv_adjust_channels_235_batch_normalization_2155_moving_variance_read_readvariableop;
7savev2_sgd_conv2d_6_kernel_momentum_read_readvariableop9
5savev2_sgd_conv2d_6_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableop<
8savev2_sgd_conv2d_27_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_27_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_27_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_27_beta_momentum_read_readvariableop<
8savev2_sgd_conv2d_35_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_35_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_35_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_35_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1936_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1936_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1936_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1936_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1968_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1968_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1968_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1968_beta_momentum_read_readvariableop=
9savev2_sgd_conv2d_163_kernel_momentum_read_readvariableop;
7savev2_sgd_conv2d_163_bias_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_163_gamma_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_163_beta_momentum_read_readvariableop=
9savev2_sgd_conv2d_195_kernel_momentum_read_readvariableop;
7savev2_sgd_conv2d_195_bias_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_195_gamma_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_195_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1522_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1522_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1522_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1522_beta_momentum_read_readvariableop>
:savev2_sgd_conv2d_1556_kernel_momentum_read_readvariableop<
8savev2_sgd_conv2d_1556_bias_momentum_read_readvariableopJ
Fsavev2_sgd_batch_normalization_1556_gamma_momentum_read_readvariableopI
Esavev2_sgd_batch_normalization_1556_beta_momentum_read_readvariableop
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
ShardedFilename?8
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*?7
value?7B?7iB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*?
value?B?iB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop-savev2_conv2d_1936_kernel_read_readvariableop+savev2_conv2d_1936_bias_read_readvariableop9savev2_batch_normalization_1936_gamma_read_readvariableop8savev2_batch_normalization_1936_beta_read_readvariableop?savev2_batch_normalization_1936_moving_mean_read_readvariableopCsavev2_batch_normalization_1936_moving_variance_read_readvariableop-savev2_conv2d_1968_kernel_read_readvariableop+savev2_conv2d_1968_bias_read_readvariableop9savev2_batch_normalization_1968_gamma_read_readvariableop8savev2_batch_normalization_1968_beta_read_readvariableop?savev2_batch_normalization_1968_moving_mean_read_readvariableopCsavev2_batch_normalization_1968_moving_variance_read_readvariableop,savev2_conv2d_163_kernel_read_readvariableop*savev2_conv2d_163_bias_read_readvariableop8savev2_batch_normalization_163_gamma_read_readvariableop7savev2_batch_normalization_163_beta_read_readvariableop>savev2_batch_normalization_163_moving_mean_read_readvariableopBsavev2_batch_normalization_163_moving_variance_read_readvariableop,savev2_conv2d_195_kernel_read_readvariableop*savev2_conv2d_195_bias_read_readvariableop8savev2_batch_normalization_195_gamma_read_readvariableop7savev2_batch_normalization_195_beta_read_readvariableop>savev2_batch_normalization_195_moving_mean_read_readvariableopBsavev2_batch_normalization_195_moving_variance_read_readvariableop-savev2_conv2d_1522_kernel_read_readvariableop+savev2_conv2d_1522_bias_read_readvariableop9savev2_batch_normalization_1522_gamma_read_readvariableop8savev2_batch_normalization_1522_beta_read_readvariableop?savev2_batch_normalization_1522_moving_mean_read_readvariableopCsavev2_batch_normalization_1522_moving_variance_read_readvariableop-savev2_conv2d_1556_kernel_read_readvariableop+savev2_conv2d_1556_bias_read_readvariableop9savev2_batch_normalization_1556_gamma_read_readvariableop8savev2_batch_normalization_1556_beta_read_readvariableop?savev2_batch_normalization_1556_moving_mean_read_readvariableopCsavev2_batch_normalization_1556_moving_variance_read_readvariableop+savev2_dense_630_kernel_read_readvariableop)savev2_dense_630_bias_read_readvariableop+savev2_dense_631_kernel_read_readvariableop)savev2_dense_631_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableopFsavev2_conv_adjust_channels_235_conv2d_2155_kernel_read_readvariableopDsavev2_conv_adjust_channels_235_conv2d_2155_bias_read_readvariableopRsavev2_conv_adjust_channels_235_batch_normalization_2155_gamma_read_readvariableopQsavev2_conv_adjust_channels_235_batch_normalization_2155_beta_read_readvariableopXsavev2_conv_adjust_channels_235_batch_normalization_2155_moving_mean_read_readvariableop\savev2_conv_adjust_channels_235_batch_normalization_2155_moving_variance_read_readvariableop7savev2_sgd_conv2d_6_kernel_momentum_read_readvariableop5savev2_sgd_conv2d_6_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableop8savev2_sgd_conv2d_27_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_27_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_27_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_27_beta_momentum_read_readvariableop8savev2_sgd_conv2d_35_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_35_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_35_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_35_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1936_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1936_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1936_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1936_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1968_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1968_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1968_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1968_beta_momentum_read_readvariableop9savev2_sgd_conv2d_163_kernel_momentum_read_readvariableop7savev2_sgd_conv2d_163_bias_momentum_read_readvariableopEsavev2_sgd_batch_normalization_163_gamma_momentum_read_readvariableopDsavev2_sgd_batch_normalization_163_beta_momentum_read_readvariableop9savev2_sgd_conv2d_195_kernel_momentum_read_readvariableop7savev2_sgd_conv2d_195_bias_momentum_read_readvariableopEsavev2_sgd_batch_normalization_195_gamma_momentum_read_readvariableopDsavev2_sgd_batch_normalization_195_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1522_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1522_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1522_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1522_beta_momentum_read_readvariableop:savev2_sgd_conv2d_1556_kernel_momentum_read_readvariableop8savev2_sgd_conv2d_1556_bias_momentum_read_readvariableopFsavev2_sgd_batch_normalization_1556_gamma_momentum_read_readvariableopEsavev2_sgd_batch_normalization_1556_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *w
dtypesm
k2i	2
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
?: ::::::::::::::::::: : : : : : :  : : : : : : : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:???:?:	?
:
: : : : : :::::::::::::::::: : : : :  : : : : : : : :  : : : : @:@:@:@:@@:@:@:@: 2(
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
:: 
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
: : 
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
: :  
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
: @: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@:'7#
!
_output_shapes
:???:!8

_output_shapes	
:?:%9!

_output_shapes
:	?
: :

_output_shapes
:
:;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :,?(
&
_output_shapes
: : @

_output_shapes
:: A

_output_shapes
:: B
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
:: F
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
:: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
::,M(
&
_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
::,Q(
&
_output_shapes
: : R
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
: : Z

_output_shapes
: : [

_output_shapes
: : \

_output_shapes
: :,](
&
_output_shapes
:  : ^

_output_shapes
: : _

_output_shapes
: : `

_output_shapes
: :,a(
&
_output_shapes
: @: b

_output_shapes
:@: c

_output_shapes
:@: d

_output_shapes
:@:,e(
&
_output_shapes
:@@: f

_output_shapes
:@: g

_output_shapes
:@: h

_output_shapes
:@:i

_output_shapes
: 
?
?
9__inference_batch_normalization_163_layer_call_fn_7477511

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_74743462
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
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7473948

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
?
?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7474154

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
?
?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477642

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
?
?
+__inference_model_314_layer_call_fn_7476350

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

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity??StatefulPartitionedCall?	
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*N
_read_only_resource_inputs0
.,	
 !"%&'(+,-.1234789:=>?@*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_314_layer_call_and_return_conditional_losses_74751562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7472449

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
?
?
:__inference_batch_normalization_2155_layer_call_fn_7478202

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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_74730122
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
?
?
8__inference_batch_normalization_35_layer_call_fn_7476867

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_74726692
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
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477485

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
?
?
8__inference_batch_normalization_27_layer_call_fn_7476774

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_74725652
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7473723

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
?
?
+__inference_model_314_layer_call_fn_7475588
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

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity??StatefulPartitionedCall?	
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_314_layer_call_and_return_conditional_losses_74754572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_7477121

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
?
a
E__inference_re_lu_27_layer_call_and_return_conditional_losses_7476792

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
-__inference_conv2d_1968_layer_call_fn_7477130

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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_74741192
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
?
E
)__inference_re_lu_6_layer_call_fn_7476640

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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_74737642
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
?
?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477886

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_7477263

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477150

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
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7473090

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
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7476993

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
?
?
:__inference_conv_adjust_channels_235_layer_call_fn_7477335

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_74731662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_1522_layer_call_fn_7477773

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
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_74735172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_model_314_layer_call_fn_7476483

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

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity??StatefulPartitionedCall?	
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_314_layer_call_and_return_conditional_losses_74754572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476743

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
F__inference_dense_631_layer_call_and_return_conditional_losses_7474800

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
?
?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477232

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
?
?
8__inference_batch_normalization_35_layer_call_fn_7476880

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_74727002
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
?
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473127
input_1
conv2d_2155_7473048
conv2d_2155_7473050$
 batch_normalization_2155_7473117$
 batch_normalization_2155_7473119$
 batch_normalization_2155_7473121$
 batch_normalization_2155_7473123
identity??0batch_normalization_2155/StatefulPartitionedCall?#conv2d_2155/StatefulPartitionedCall?
#conv2d_2155/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_2155_7473048conv2d_2155_7473050*
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
GPU 2J 8? *Q
fLRJ
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_74730372%
#conv2d_2155/StatefulPartitionedCall?
0batch_normalization_2155/StatefulPartitionedCallStatefulPartitionedCall,conv2d_2155/StatefulPartitionedCall:output:0 batch_normalization_2155_7473117 batch_normalization_2155_7473119 batch_normalization_2155_7473121 batch_normalization_2155_7473123*
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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_747307222
0batch_normalization_2155/StatefulPartitionedCall?
IdentityIdentity9batch_normalization_2155/StatefulPartitionedCall:output:01^batch_normalization_2155/StatefulPartitionedCall$^conv2d_2155/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2d
0batch_normalization_2155/StatefulPartitionedCall0batch_normalization_2155/StatefulPartitionedCall2J
#conv2d_2155/StatefulPartitionedCall#conv2d_2155/StatefulPartitionedCall:X T
/
_output_shapes
:?????????   
!
_user_specified_name	input_1
ڷ
?
F__inference_model_314_layer_call_and_return_conditional_losses_7474817
input_1
conv2d_6_7473681
conv2d_6_7473683!
batch_normalization_6_7473750!
batch_normalization_6_7473752!
batch_normalization_6_7473754!
batch_normalization_6_7473756
conv2d_27_7473794
conv2d_27_7473796"
batch_normalization_27_7473863"
batch_normalization_27_7473865"
batch_normalization_27_7473867"
batch_normalization_27_7473869
conv2d_35_7473906
conv2d_35_7473908"
batch_normalization_35_7473975"
batch_normalization_35_7473977"
batch_normalization_35_7473979"
batch_normalization_35_7473981
conv2d_1936_7474018
conv2d_1936_7474020$
 batch_normalization_1936_7474087$
 batch_normalization_1936_7474089$
 batch_normalization_1936_7474091$
 batch_normalization_1936_7474093
conv2d_1968_7474130
conv2d_1968_7474132$
 batch_normalization_1968_7474199$
 batch_normalization_1968_7474201$
 batch_normalization_1968_7474203$
 batch_normalization_1968_7474205$
 conv_adjust_channels_235_7474255$
 conv_adjust_channels_235_7474257$
 conv_adjust_channels_235_7474259$
 conv_adjust_channels_235_7474261$
 conv_adjust_channels_235_7474263$
 conv_adjust_channels_235_7474265
conv2d_163_7474304
conv2d_163_7474306#
batch_normalization_163_7474373#
batch_normalization_163_7474375#
batch_normalization_163_7474377#
batch_normalization_163_7474379
conv2d_195_7474416
conv2d_195_7474418#
batch_normalization_195_7474485#
batch_normalization_195_7474487#
batch_normalization_195_7474489#
batch_normalization_195_7474491
conv2d_1522_7474543
conv2d_1522_7474545$
 batch_normalization_1522_7474612$
 batch_normalization_1522_7474614$
 batch_normalization_1522_7474616$
 batch_normalization_1522_7474618
conv2d_1556_7474656
conv2d_1556_7474658$
 batch_normalization_1556_7474725$
 batch_normalization_1556_7474727$
 batch_normalization_1556_7474729$
 batch_normalization_1556_7474731
dense_630_7474784
dense_630_7474786
dense_631_7474811
dense_631_7474813
identity??0batch_normalization_1522/StatefulPartitionedCall?0batch_normalization_1556/StatefulPartitionedCall?/batch_normalization_163/StatefulPartitionedCall?0batch_normalization_1936/StatefulPartitionedCall?/batch_normalization_195/StatefulPartitionedCall?0batch_normalization_1968/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?#conv2d_1522/StatefulPartitionedCall?#conv2d_1556/StatefulPartitionedCall?"conv2d_163/StatefulPartitionedCall?#conv2d_1936/StatefulPartitionedCall?"conv2d_195/StatefulPartitionedCall?#conv2d_1968/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?0conv_adjust_channels_235/StatefulPartitionedCall?!dense_630/StatefulPartitionedCall?!dense_631/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_6_7473681conv2d_6_7473683*
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_74736702"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_7473750batch_normalization_6_7473752batch_normalization_6_7473754batch_normalization_6_7473756*
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74737052/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_74737642
re_lu_6/PartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_74724972!
max_pooling2d_6/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_27_7473794conv2d_27_7473796*
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_74737832#
!conv2d_27/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_27_7473863batch_normalization_27_7473865batch_normalization_27_7473867batch_normalization_27_7473869*
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_747381820
.batch_normalization_27/StatefulPartitionedCall?
re_lu_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_74738772
re_lu_27/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall!re_lu_27/PartitionedCall:output:0conv2d_35_7473906conv2d_35_7473908*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_74738952#
!conv2d_35/StatefulPartitionedCall?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_7473975batch_normalization_35_7473977batch_normalization_35_7473979batch_normalization_35_7473981*
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_747393020
.batch_normalization_35/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_74739892
re_lu_35/PartitionedCall?
#conv2d_1936/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0conv2d_1936_7474018conv2d_1936_7474020*
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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_74740072%
#conv2d_1936/StatefulPartitionedCall?
0batch_normalization_1936/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1936/StatefulPartitionedCall:output:0 batch_normalization_1936_7474087 batch_normalization_1936_7474089 batch_normalization_1936_7474091 batch_normalization_1936_7474093*
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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_747404222
0batch_normalization_1936/StatefulPartitionedCall?
re_lu_1763/PartitionedCallPartitionedCall9batch_normalization_1936/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_74741012
re_lu_1763/PartitionedCall?
#conv2d_1968/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1763/PartitionedCall:output:0conv2d_1968_7474130conv2d_1968_7474132*
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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_74741192%
#conv2d_1968/StatefulPartitionedCall?
0batch_normalization_1968/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1968/StatefulPartitionedCall:output:0 batch_normalization_1968_7474199 batch_normalization_1968_7474201 batch_normalization_1968_7474203 batch_normalization_1968_7474205*
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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_747415422
0batch_normalization_1968/StatefulPartitionedCall?
re_lu_1795/PartitionedCallPartitionedCall9batch_normalization_1968/StatefulPartitionedCall:output:0*
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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_74742132
re_lu_1795/PartitionedCall?
0conv_adjust_channels_235/StatefulPartitionedCallStatefulPartitionedCall#re_lu_1795/PartitionedCall:output:0 conv_adjust_channels_235_7474255 conv_adjust_channels_235_7474257 conv_adjust_channels_235_7474259 conv_adjust_channels_235_7474261 conv_adjust_channels_235_7474263 conv_adjust_channels_235_7474265*
Tin
	2*
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
GPU 2J 8? *^
fYRW
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_747316622
0conv_adjust_channels_235/StatefulPartitionedCall?
add_356/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:09conv_adjust_channels_235/StatefulPartitionedCall:output:0*
Tin
2*
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
D__inference_add_356_layer_call_and_return_conditional_losses_74742742
add_356/PartitionedCall?
"conv2d_163/StatefulPartitionedCallStatefulPartitionedCall add_356/PartitionedCall:output:0conv2d_163_7474304conv2d_163_7474306*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_163_layer_call_and_return_conditional_losses_74742932$
"conv2d_163/StatefulPartitionedCall?
/batch_normalization_163/StatefulPartitionedCallStatefulPartitionedCall+conv2d_163/StatefulPartitionedCall:output:0batch_normalization_163_7474373batch_normalization_163_7474375batch_normalization_163_7474377batch_normalization_163_7474379*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_747432821
/batch_normalization_163/StatefulPartitionedCall?
re_lu_163/PartitionedCallPartitionedCall8batch_normalization_163/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_163_layer_call_and_return_conditional_losses_74743872
re_lu_163/PartitionedCall?
"conv2d_195/StatefulPartitionedCallStatefulPartitionedCall"re_lu_163/PartitionedCall:output:0conv2d_195_7474416conv2d_195_7474418*
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
GPU 2J 8? *P
fKRI
G__inference_conv2d_195_layer_call_and_return_conditional_losses_74744052$
"conv2d_195/StatefulPartitionedCall?
/batch_normalization_195/StatefulPartitionedCallStatefulPartitionedCall+conv2d_195/StatefulPartitionedCall:output:0batch_normalization_195_7474485batch_normalization_195_7474487batch_normalization_195_7474489batch_normalization_195_7474491*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_747444021
/batch_normalization_195/StatefulPartitionedCall?
re_lu_195/PartitionedCallPartitionedCall8batch_normalization_195/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_re_lu_195_layer_call_and_return_conditional_losses_74744992
re_lu_195/PartitionedCall?
add_357/PartitionedCallPartitionedCall"re_lu_195/PartitionedCall:output:0#re_lu_1795/PartitionedCall:output:0*
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
D__inference_add_357_layer_call_and_return_conditional_losses_74745132
add_357/PartitionedCall?
#conv2d_1522/StatefulPartitionedCallStatefulPartitionedCall add_357/PartitionedCall:output:0conv2d_1522_7474543conv2d_1522_7474545*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_74745322%
#conv2d_1522/StatefulPartitionedCall?
0batch_normalization_1522/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1522/StatefulPartitionedCall:output:0 batch_normalization_1522_7474612 batch_normalization_1522_7474614 batch_normalization_1522_7474616 batch_normalization_1522_7474618*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_747456722
0batch_normalization_1522/StatefulPartitionedCall?
re_lu_1404/PartitionedCallPartitionedCall9batch_normalization_1522/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_74746262
re_lu_1404/PartitionedCall?
"max_pooling2d_1404/PartitionedCallPartitionedCall#re_lu_1404/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_74735342$
"max_pooling2d_1404/PartitionedCall?
#conv2d_1556/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_1404/PartitionedCall:output:0conv2d_1556_7474656conv2d_1556_7474658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_74746452%
#conv2d_1556/StatefulPartitionedCall?
0batch_normalization_1556/StatefulPartitionedCallStatefulPartitionedCall,conv2d_1556/StatefulPartitionedCall:output:0 batch_normalization_1556_7474725 batch_normalization_1556_7474727 batch_normalization_1556_7474729 batch_normalization_1556_7474731*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_747468022
0batch_normalization_1556/StatefulPartitionedCall?
re_lu_1438/PartitionedCallPartitionedCall9batch_normalization_1556/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_74747392
re_lu_1438/PartitionedCall?
"max_pooling2d_1438/PartitionedCallPartitionedCall#re_lu_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_74736502$
"max_pooling2d_1438/PartitionedCall?
flatten_315/PartitionedCallPartitionedCall+max_pooling2d_1438/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_315_layer_call_and_return_conditional_losses_74747542
flatten_315/PartitionedCall?
!dense_630/StatefulPartitionedCallStatefulPartitionedCall$flatten_315/PartitionedCall:output:0dense_630_7474784dense_630_7474786*
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
F__inference_dense_630_layer_call_and_return_conditional_losses_74747732#
!dense_630/StatefulPartitionedCall?
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_7474811dense_631_7474813*
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
F__inference_dense_631_layer_call_and_return_conditional_losses_74748002#
!dense_631/StatefulPartitionedCall?
IdentityIdentity*dense_631/StatefulPartitionedCall:output:01^batch_normalization_1522/StatefulPartitionedCall1^batch_normalization_1556/StatefulPartitionedCall0^batch_normalization_163/StatefulPartitionedCall1^batch_normalization_1936/StatefulPartitionedCall0^batch_normalization_195/StatefulPartitionedCall1^batch_normalization_1968/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall$^conv2d_1522/StatefulPartitionedCall$^conv2d_1556/StatefulPartitionedCall#^conv2d_163/StatefulPartitionedCall$^conv2d_1936/StatefulPartitionedCall#^conv2d_195/StatefulPartitionedCall$^conv2d_1968/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall1^conv_adjust_channels_235/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2d
0batch_normalization_1522/StatefulPartitionedCall0batch_normalization_1522/StatefulPartitionedCall2d
0batch_normalization_1556/StatefulPartitionedCall0batch_normalization_1556/StatefulPartitionedCall2b
/batch_normalization_163/StatefulPartitionedCall/batch_normalization_163/StatefulPartitionedCall2d
0batch_normalization_1936/StatefulPartitionedCall0batch_normalization_1936/StatefulPartitionedCall2b
/batch_normalization_195/StatefulPartitionedCall/batch_normalization_195/StatefulPartitionedCall2d
0batch_normalization_1968/StatefulPartitionedCall0batch_normalization_1968/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2J
#conv2d_1522/StatefulPartitionedCall#conv2d_1522/StatefulPartitionedCall2J
#conv2d_1556/StatefulPartitionedCall#conv2d_1556/StatefulPartitionedCall2H
"conv2d_163/StatefulPartitionedCall"conv2d_163/StatefulPartitionedCall2J
#conv2d_1936/StatefulPartitionedCall#conv2d_1936/StatefulPartitionedCall2H
"conv2d_195/StatefulPartitionedCall"conv2d_195/StatefulPartitionedCall2J
#conv2d_1968/StatefulPartitionedCall#conv2d_1968/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2d
0conv_adjust_channels_235/StatefulPartitionedCall0conv_adjust_channels_235/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
H
,__inference_re_lu_1795_layer_call_fn_7477268

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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_74742132
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
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7473072

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476918

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
?	
?
G__inference_conv2d_163_layer_call_and_return_conditional_losses_7477374

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
?
?
+__inference_dense_631_layer_call_fn_7478055

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
F__inference_dense_631_layer_call_and_return_conditional_losses_74748002
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
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476540

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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477168

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
?
c
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_7477999

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_6_layer_call_fn_7476566

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_74724802
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
?
?
:__inference_batch_normalization_2155_layer_call_fn_7478189

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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_74729812
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
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477075

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7472669

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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_7474007

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
?	
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_7473670

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
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7473930

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
??
?@
#__inference__traced_restore_7478859
file_prefix$
 assignvariableop_conv2d_6_kernel$
 assignvariableop_1_conv2d_6_bias2
.assignvariableop_2_batch_normalization_6_gamma1
-assignvariableop_3_batch_normalization_6_beta8
4assignvariableop_4_batch_normalization_6_moving_mean<
8assignvariableop_5_batch_normalization_6_moving_variance'
#assignvariableop_6_conv2d_27_kernel%
!assignvariableop_7_conv2d_27_bias3
/assignvariableop_8_batch_normalization_27_gamma2
.assignvariableop_9_batch_normalization_27_beta:
6assignvariableop_10_batch_normalization_27_moving_mean>
:assignvariableop_11_batch_normalization_27_moving_variance(
$assignvariableop_12_conv2d_35_kernel&
"assignvariableop_13_conv2d_35_bias4
0assignvariableop_14_batch_normalization_35_gamma3
/assignvariableop_15_batch_normalization_35_beta:
6assignvariableop_16_batch_normalization_35_moving_mean>
:assignvariableop_17_batch_normalization_35_moving_variance*
&assignvariableop_18_conv2d_1936_kernel(
$assignvariableop_19_conv2d_1936_bias6
2assignvariableop_20_batch_normalization_1936_gamma5
1assignvariableop_21_batch_normalization_1936_beta<
8assignvariableop_22_batch_normalization_1936_moving_mean@
<assignvariableop_23_batch_normalization_1936_moving_variance*
&assignvariableop_24_conv2d_1968_kernel(
$assignvariableop_25_conv2d_1968_bias6
2assignvariableop_26_batch_normalization_1968_gamma5
1assignvariableop_27_batch_normalization_1968_beta<
8assignvariableop_28_batch_normalization_1968_moving_mean@
<assignvariableop_29_batch_normalization_1968_moving_variance)
%assignvariableop_30_conv2d_163_kernel'
#assignvariableop_31_conv2d_163_bias5
1assignvariableop_32_batch_normalization_163_gamma4
0assignvariableop_33_batch_normalization_163_beta;
7assignvariableop_34_batch_normalization_163_moving_mean?
;assignvariableop_35_batch_normalization_163_moving_variance)
%assignvariableop_36_conv2d_195_kernel'
#assignvariableop_37_conv2d_195_bias5
1assignvariableop_38_batch_normalization_195_gamma4
0assignvariableop_39_batch_normalization_195_beta;
7assignvariableop_40_batch_normalization_195_moving_mean?
;assignvariableop_41_batch_normalization_195_moving_variance*
&assignvariableop_42_conv2d_1522_kernel(
$assignvariableop_43_conv2d_1522_bias6
2assignvariableop_44_batch_normalization_1522_gamma5
1assignvariableop_45_batch_normalization_1522_beta<
8assignvariableop_46_batch_normalization_1522_moving_mean@
<assignvariableop_47_batch_normalization_1522_moving_variance*
&assignvariableop_48_conv2d_1556_kernel(
$assignvariableop_49_conv2d_1556_bias6
2assignvariableop_50_batch_normalization_1556_gamma5
1assignvariableop_51_batch_normalization_1556_beta<
8assignvariableop_52_batch_normalization_1556_moving_mean@
<assignvariableop_53_batch_normalization_1556_moving_variance(
$assignvariableop_54_dense_630_kernel&
"assignvariableop_55_dense_630_bias(
$assignvariableop_56_dense_631_kernel&
"assignvariableop_57_dense_631_bias 
assignvariableop_58_sgd_iter!
assignvariableop_59_sgd_decay)
%assignvariableop_60_sgd_learning_rate$
 assignvariableop_61_sgd_momentumC
?assignvariableop_62_conv_adjust_channels_235_conv2d_2155_kernelA
=assignvariableop_63_conv_adjust_channels_235_conv2d_2155_biasO
Kassignvariableop_64_conv_adjust_channels_235_batch_normalization_2155_gammaN
Jassignvariableop_65_conv_adjust_channels_235_batch_normalization_2155_betaU
Qassignvariableop_66_conv_adjust_channels_235_batch_normalization_2155_moving_meanY
Uassignvariableop_67_conv_adjust_channels_235_batch_normalization_2155_moving_variance4
0assignvariableop_68_sgd_conv2d_6_kernel_momentum2
.assignvariableop_69_sgd_conv2d_6_bias_momentum@
<assignvariableop_70_sgd_batch_normalization_6_gamma_momentum?
;assignvariableop_71_sgd_batch_normalization_6_beta_momentum5
1assignvariableop_72_sgd_conv2d_27_kernel_momentum3
/assignvariableop_73_sgd_conv2d_27_bias_momentumA
=assignvariableop_74_sgd_batch_normalization_27_gamma_momentum@
<assignvariableop_75_sgd_batch_normalization_27_beta_momentum5
1assignvariableop_76_sgd_conv2d_35_kernel_momentum3
/assignvariableop_77_sgd_conv2d_35_bias_momentumA
=assignvariableop_78_sgd_batch_normalization_35_gamma_momentum@
<assignvariableop_79_sgd_batch_normalization_35_beta_momentum7
3assignvariableop_80_sgd_conv2d_1936_kernel_momentum5
1assignvariableop_81_sgd_conv2d_1936_bias_momentumC
?assignvariableop_82_sgd_batch_normalization_1936_gamma_momentumB
>assignvariableop_83_sgd_batch_normalization_1936_beta_momentum7
3assignvariableop_84_sgd_conv2d_1968_kernel_momentum5
1assignvariableop_85_sgd_conv2d_1968_bias_momentumC
?assignvariableop_86_sgd_batch_normalization_1968_gamma_momentumB
>assignvariableop_87_sgd_batch_normalization_1968_beta_momentum6
2assignvariableop_88_sgd_conv2d_163_kernel_momentum4
0assignvariableop_89_sgd_conv2d_163_bias_momentumB
>assignvariableop_90_sgd_batch_normalization_163_gamma_momentumA
=assignvariableop_91_sgd_batch_normalization_163_beta_momentum6
2assignvariableop_92_sgd_conv2d_195_kernel_momentum4
0assignvariableop_93_sgd_conv2d_195_bias_momentumB
>assignvariableop_94_sgd_batch_normalization_195_gamma_momentumA
=assignvariableop_95_sgd_batch_normalization_195_beta_momentum7
3assignvariableop_96_sgd_conv2d_1522_kernel_momentum5
1assignvariableop_97_sgd_conv2d_1522_bias_momentumC
?assignvariableop_98_sgd_batch_normalization_1522_gamma_momentumB
>assignvariableop_99_sgd_batch_normalization_1522_beta_momentum8
4assignvariableop_100_sgd_conv2d_1556_kernel_momentum6
2assignvariableop_101_sgd_conv2d_1556_bias_momentumD
@assignvariableop_102_sgd_batch_normalization_1556_gamma_momentumC
?assignvariableop_103_sgd_batch_normalization_1556_beta_momentum
identity_105??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?8
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*?7
value?7B?7iB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*?
value?B?iB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*w
dtypesm
k2i	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_6_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_6_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_27_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_27_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_27_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_27_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_35_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_35_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_35_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_35_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_35_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_35_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_conv2d_1936_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_1936_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp2assignvariableop_20_batch_normalization_1936_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_batch_normalization_1936_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp8assignvariableop_22_batch_normalization_1936_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp<assignvariableop_23_batch_normalization_1936_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_conv2d_1968_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_1968_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_batch_normalization_1968_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_batch_normalization_1968_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp8assignvariableop_28_batch_normalization_1968_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp<assignvariableop_29_batch_normalization_1968_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv2d_163_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv2d_163_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp1assignvariableop_32_batch_normalization_163_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_batch_normalization_163_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_batch_normalization_163_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp;assignvariableop_35_batch_normalization_163_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_conv2d_195_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp#assignvariableop_37_conv2d_195_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp1assignvariableop_38_batch_normalization_195_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp0assignvariableop_39_batch_normalization_195_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp7assignvariableop_40_batch_normalization_195_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp;assignvariableop_41_batch_normalization_195_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_conv2d_1522_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp$assignvariableop_43_conv2d_1522_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_batch_normalization_1522_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp1assignvariableop_45_batch_normalization_1522_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp8assignvariableop_46_batch_normalization_1522_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp<assignvariableop_47_batch_normalization_1522_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_conv2d_1556_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp$assignvariableop_49_conv2d_1556_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_batch_normalization_1556_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp1assignvariableop_51_batch_normalization_1556_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp8assignvariableop_52_batch_normalization_1556_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp<assignvariableop_53_batch_normalization_1556_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp$assignvariableop_54_dense_630_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp"assignvariableop_55_dense_630_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_dense_631_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp"assignvariableop_57_dense_631_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_sgd_iterIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_sgd_decayIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp%assignvariableop_60_sgd_learning_rateIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp assignvariableop_61_sgd_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp?assignvariableop_62_conv_adjust_channels_235_conv2d_2155_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp=assignvariableop_63_conv_adjust_channels_235_conv2d_2155_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpKassignvariableop_64_conv_adjust_channels_235_batch_normalization_2155_gammaIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpJassignvariableop_65_conv_adjust_channels_235_batch_normalization_2155_betaIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpQassignvariableop_66_conv_adjust_channels_235_batch_normalization_2155_moving_meanIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpUassignvariableop_67_conv_adjust_channels_235_batch_normalization_2155_moving_varianceIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_conv2d_6_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_conv2d_6_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp<assignvariableop_70_sgd_batch_normalization_6_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp;assignvariableop_71_sgd_batch_normalization_6_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp1assignvariableop_72_sgd_conv2d_27_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp/assignvariableop_73_sgd_conv2d_27_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp=assignvariableop_74_sgd_batch_normalization_27_gamma_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp<assignvariableop_75_sgd_batch_normalization_27_beta_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp1assignvariableop_76_sgd_conv2d_35_kernel_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp/assignvariableop_77_sgd_conv2d_35_bias_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp=assignvariableop_78_sgd_batch_normalization_35_gamma_momentumIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp<assignvariableop_79_sgd_batch_normalization_35_beta_momentumIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_sgd_conv2d_1936_kernel_momentumIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp1assignvariableop_81_sgd_conv2d_1936_bias_momentumIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp?assignvariableop_82_sgd_batch_normalization_1936_gamma_momentumIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp>assignvariableop_83_sgd_batch_normalization_1936_beta_momentumIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp3assignvariableop_84_sgd_conv2d_1968_kernel_momentumIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp1assignvariableop_85_sgd_conv2d_1968_bias_momentumIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp?assignvariableop_86_sgd_batch_normalization_1968_gamma_momentumIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp>assignvariableop_87_sgd_batch_normalization_1968_beta_momentumIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp2assignvariableop_88_sgd_conv2d_163_kernel_momentumIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp0assignvariableop_89_sgd_conv2d_163_bias_momentumIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp>assignvariableop_90_sgd_batch_normalization_163_gamma_momentumIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp=assignvariableop_91_sgd_batch_normalization_163_beta_momentumIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp2assignvariableop_92_sgd_conv2d_195_kernel_momentumIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp0assignvariableop_93_sgd_conv2d_195_bias_momentumIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp>assignvariableop_94_sgd_batch_normalization_195_gamma_momentumIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp=assignvariableop_95_sgd_batch_normalization_195_beta_momentumIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp3assignvariableop_96_sgd_conv2d_1522_kernel_momentumIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp1assignvariableop_97_sgd_conv2d_1522_bias_momentumIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp?assignvariableop_98_sgd_batch_normalization_1522_gamma_momentumIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp>assignvariableop_99_sgd_batch_normalization_1522_beta_momentumIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp4assignvariableop_100_sgd_conv2d_1556_kernel_momentumIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp2assignvariableop_101_sgd_conv2d_1556_bias_momentumIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp@assignvariableop_102_sgd_batch_normalization_1556_gamma_momentumIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp?assignvariableop_103_sgd_batch_normalization_1556_beta_momentumIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1039
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_104Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_104?
Identity_105IdentityIdentity_104:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_105"%
identity_105Identity_105:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032*
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
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7474060

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476522

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
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_7477700

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_7476964

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
?
?
:__inference_batch_normalization_1522_layer_call_fn_7477824

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
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_74745672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_7476493

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
?
?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477793

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
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
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_2155_layer_call_fn_7478125

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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_74730722
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
9__inference_batch_normalization_163_layer_call_fn_7477434

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_74732782
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
?
?
-__inference_conv2d_2155_layer_call_fn_7478074

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
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_74730372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7472480

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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477011

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
U
)__inference_add_356_layer_call_fn_7477364
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
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_356_layer_call_and_return_conditional_losses_74742742
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????  :?????????  :Y U
/
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
?
?
9__inference_batch_normalization_163_layer_call_fn_7477447

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_74733092
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
)__inference_add_357_layer_call_fn_7477690
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
D__inference_add_357_layer_call_and_return_conditional_losses_74745132
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
?
?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478158

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
?
a
E__inference_re_lu_35_layer_call_and_return_conditional_losses_7473989

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
?
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_7473764

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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476679

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
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477403

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
P
4__inference_max_pooling2d_1404_layer_call_fn_7473540

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
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_74735342
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
9__inference_batch_normalization_163_layer_call_fn_7477498

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_74743282
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
?
p
D__inference_add_356_layer_call_and_return_conditional_losses_7477358
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????  2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????  :?????????  :Y U
/
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
?
?
-__inference_conv2d_1936_layer_call_fn_7476973

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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_74740072
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
?
H
,__inference_re_lu_1438_layer_call_fn_7478004

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
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_74747392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_7477857

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_7475729
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

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity??StatefulPartitionedCall?	
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
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_74723872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473166

inputs
conv2d_2155_7473151
conv2d_2155_7473153$
 batch_normalization_2155_7473156$
 batch_normalization_2155_7473158$
 batch_normalization_2155_7473160$
 batch_normalization_2155_7473162
identity??0batch_normalization_2155/StatefulPartitionedCall?#conv2d_2155/StatefulPartitionedCall?
#conv2d_2155/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2155_7473151conv2d_2155_7473153*
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
GPU 2J 8? *Q
fLRJ
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_74730372%
#conv2d_2155/StatefulPartitionedCall?
0batch_normalization_2155/StatefulPartitionedCallStatefulPartitionedCall,conv2d_2155/StatefulPartitionedCall:output:0 batch_normalization_2155_7473156 batch_normalization_2155_7473158 batch_normalization_2155_7473160 batch_normalization_2155_7473162*
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
GPU 2J 8? *^
fYRW
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_747307222
0batch_normalization_2155/StatefulPartitionedCall?
IdentityIdentity9batch_normalization_2155/StatefulPartitionedCall:output:01^batch_normalization_2155/StatefulPartitionedCall$^conv2d_2155/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????   ::::::2d
0batch_normalization_2155/StatefulPartitionedCall0batch_normalization_2155/StatefulPartitionedCall2J
#conv2d_2155/StatefulPartitionedCall#conv2d_2155/StatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7474698

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_6_layer_call_fn_7472503

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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_74724972
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
	dense_6310
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
??
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
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer-25
layer_with_weights-15
layer-26
layer_with_weights-16
layer-27
layer-28
layer-29
layer_with_weights-17
layer-30
 layer_with_weights-18
 layer-31
!layer-32
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer_with_weights-20
%layer-36
&	optimizer
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "model_314", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_314", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_6", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_27", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_27", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["re_lu_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_35", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_35", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_35", "inbound_nodes": [[["batch_normalization_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1936", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1936", "inbound_nodes": [[["re_lu_35", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1936", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1936", "inbound_nodes": [[["conv2d_1936", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1763", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1763", "inbound_nodes": [[["batch_normalization_1936", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1968", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1968", "inbound_nodes": [[["re_lu_1763", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1968", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1968", "inbound_nodes": [[["conv2d_1968", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1795", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1795", "inbound_nodes": [[["batch_normalization_1968", 0, 0, {}]]]}, {"class_name": "ConvAdjustChannels", "config": {"layer was saved without config": true}, "name": "conv_adjust_channels_235", "inbound_nodes": [[["re_lu_1795", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_356", "trainable": true, "dtype": "float32"}, "name": "add_356", "inbound_nodes": [[["re_lu_35", 0, 0, {}], ["conv_adjust_channels_235", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_163", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_163", "inbound_nodes": [[["add_356", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_163", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_163", "inbound_nodes": [[["conv2d_163", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_163", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_163", "inbound_nodes": [[["batch_normalization_163", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_195", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_195", "inbound_nodes": [[["re_lu_163", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_195", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_195", "inbound_nodes": [[["conv2d_195", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_195", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_195", "inbound_nodes": [[["batch_normalization_195", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_357", "trainable": true, "dtype": "float32"}, "name": "add_357", "inbound_nodes": [[["re_lu_195", 0, 0, {}], ["re_lu_1795", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1522", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1522", "inbound_nodes": [[["add_357", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1522", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1522", "inbound_nodes": [[["conv2d_1522", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1404", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1404", "inbound_nodes": [[["batch_normalization_1522", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1404", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1404", "inbound_nodes": [[["re_lu_1404", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1556", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1556", "inbound_nodes": [[["max_pooling2d_1404", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1556", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1556", "inbound_nodes": [[["conv2d_1556", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1438", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1438", "inbound_nodes": [[["batch_normalization_1556", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1438", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1438", "inbound_nodes": [[["re_lu_1438", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_315", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_315", "inbound_nodes": [[["max_pooling2d_1438", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_630", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_630", "inbound_nodes": [[["flatten_315", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_631", "inbound_nodes": [[["dense_630", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_631", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
?	
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8regularization_losses
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
;trainable_variables
<regularization_losses
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?trainable_variables
@regularization_losses
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_27", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
\axis
	]gamma
^beta
_moving_mean
`moving_variance
atrainable_variables
bregularization_losses
c	variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_35", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?
etrainable_variables
fregularization_losses
g	variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1936", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1936", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1936", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1936", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1763", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1763", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	

|kernel
}bias
~trainable_variables
regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1968", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1968", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1968", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1968", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1795", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1795", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
	?conv
?bn
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ConvAdjustChannels", "name": "conv_adjust_channels_235", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ConvAdjustChannels"}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_356", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_356", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 16]}, {"class_name": "TensorShape", "items": [null, 32, 32, 16]}]}
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_163", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_163", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_163", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_163", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_163", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_163", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_195", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_195", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_195", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_195", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_195", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_195", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_357", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_357", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 32]}, {"class_name": "TensorShape", "items": [null, 32, 32, 32]}]}
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1522", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1522", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1522", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1522", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1404", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1404", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1404", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1404", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1556", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1556", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1556", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1556", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1438", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1438", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1438", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1438", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_315", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_315", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_630", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_630", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65536}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65536]}}
?
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_631", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter

?decay
?learning_rate
?momentum,momentum?-momentum?3momentum?4momentum?Cmomentum?Dmomentum?Jmomentum?Kmomentum?Vmomentum?Wmomentum?]momentum?^momentum?imomentum?jmomentum?pmomentum?qmomentum?|momentum?}momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum??momentum?"
	optimizer
?
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
i12
j13
p14
q15
|16
}17
?18
?19
?20
?21
?22
?23
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
?43"
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
i18
j19
p20
q21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
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
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63"
trackable_list_wrapper
?
 ?layer_regularization_losses
'trainable_variables
?layer_metrics
(regularization_losses
?layers
?non_trainable_variables
)	variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'2conv2d_6/kernel
:2conv2d_6/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
 ?layer_regularization_losses
.trainable_variables
?layer_metrics
?layers
/regularization_losses
?non_trainable_variables
0	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
?
 ?layer_regularization_losses
7trainable_variables
?layer_metrics
?layers
8regularization_losses
?non_trainable_variables
9	variables
?metrics
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
 ?layer_regularization_losses
;trainable_variables
?layer_metrics
?layers
<regularization_losses
?non_trainable_variables
=	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
@regularization_losses
?non_trainable_variables
A	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_27/kernel
:2conv2d_27/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Etrainable_variables
?layer_metrics
?layers
Fregularization_losses
?non_trainable_variables
G	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_27/gamma
):'2batch_normalization_27/beta
2:0 (2"batch_normalization_27/moving_mean
6:4 (2&batch_normalization_27/moving_variance
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
?
 ?layer_regularization_losses
Ntrainable_variables
?layer_metrics
?layers
Oregularization_losses
?non_trainable_variables
P	variables
?metrics
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
 ?layer_regularization_losses
Rtrainable_variables
?layer_metrics
?layers
Sregularization_losses
?non_trainable_variables
T	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_35/kernel
:2conv2d_35/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Xtrainable_variables
?layer_metrics
?layers
Yregularization_losses
?non_trainable_variables
Z	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_35/gamma
):'2batch_normalization_35/beta
2:0 (2"batch_normalization_35/moving_mean
6:4 (2&batch_normalization_35/moving_variance
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
?
 ?layer_regularization_losses
atrainable_variables
?layer_metrics
?layers
bregularization_losses
?non_trainable_variables
c	variables
?metrics
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
 ?layer_regularization_losses
etrainable_variables
?layer_metrics
?layers
fregularization_losses
?non_trainable_variables
g	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* 2conv2d_1936/kernel
: 2conv2d_1936/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
 ?layer_regularization_losses
ktrainable_variables
?layer_metrics
?layers
lregularization_losses
?non_trainable_variables
m	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1936/gamma
+:) 2batch_normalization_1936/beta
4:2  (2$batch_normalization_1936/moving_mean
8:6  (2(batch_normalization_1936/moving_variance
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
?
 ?layer_regularization_losses
ttrainable_variables
?layer_metrics
?layers
uregularization_losses
?non_trainable_variables
v	variables
?metrics
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
 ?layer_regularization_losses
xtrainable_variables
?layer_metrics
?layers
yregularization_losses
?non_trainable_variables
z	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1968/kernel
: 2conv2d_1968/bias
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
?
 ?layer_regularization_losses
~trainable_variables
?layer_metrics
?layers
regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1968/gamma
+:) 2batch_normalization_1968/beta
4:2  (2$batch_normalization_1968/moving_mean
8:6  (2(batch_normalization_1968/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2155", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2155", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2155", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2155", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv2d_163/kernel
: 2conv2d_163/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_163/gamma
*:( 2batch_normalization_163/beta
3:1  (2#batch_normalization_163/moving_mean
7:5  (2'batch_normalization_163/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)  2conv2d_195/kernel
: 2conv2d_195/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_195/gamma
*:( 2batch_normalization_195/beta
3:1  (2#batch_normalization_195/moving_mean
7:5  (2'batch_normalization_195/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* @2conv2d_1522/kernel
:@2conv2d_1522/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*@2batch_normalization_1522/gamma
+:)@2batch_normalization_1522/beta
4:2@ (2$batch_normalization_1522/moving_mean
8:6@ (2(batch_normalization_1522/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*@@2conv2d_1556/kernel
:@2conv2d_1556/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*@2batch_normalization_1556/gamma
+:)@2batch_normalization_1556/beta
4:2@ (2$batch_normalization_1556/moving_mean
8:6@ (2(batch_normalization_1556/moving_variance
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
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
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#???2dense_630/kernel
:?2dense_630/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?
2dense_631/kernel
:
2dense_631/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
E:C 2+conv_adjust_channels_235/conv2d_2155/kernel
7:52)conv_adjust_channels_235/conv2d_2155/bias
E:C27conv_adjust_channels_235/batch_normalization_2155/gamma
D:B26conv_adjust_channels_235/batch_normalization_2155/beta
M:K (2=conv_adjust_channels_235/batch_normalization_2155/moving_mean
Q:O (2Aconv_adjust_channels_235/batch_normalization_2155/moving_variance
 "
trackable_list_wrapper
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
?
50
61
L2
M3
_4
`5
r6
s7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
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
.
50
61"
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
.
L0
M1"
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
.
_0
`1"
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
.
r0
s1"
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
0
?0
?1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
4:22SGD/conv2d_6/kernel/momentum
&:$2SGD/conv2d_6/bias/momentum
4:22(SGD/batch_normalization_6/gamma/momentum
3:12'SGD/batch_normalization_6/beta/momentum
5:32SGD/conv2d_27/kernel/momentum
':%2SGD/conv2d_27/bias/momentum
5:32)SGD/batch_normalization_27/gamma/momentum
4:22(SGD/batch_normalization_27/beta/momentum
5:32SGD/conv2d_35/kernel/momentum
':%2SGD/conv2d_35/bias/momentum
5:32)SGD/batch_normalization_35/gamma/momentum
4:22(SGD/batch_normalization_35/beta/momentum
7:5 2SGD/conv2d_1936/kernel/momentum
):' 2SGD/conv2d_1936/bias/momentum
7:5 2+SGD/batch_normalization_1936/gamma/momentum
6:4 2*SGD/batch_normalization_1936/beta/momentum
7:5  2SGD/conv2d_1968/kernel/momentum
):' 2SGD/conv2d_1968/bias/momentum
7:5 2+SGD/batch_normalization_1968/gamma/momentum
6:4 2*SGD/batch_normalization_1968/beta/momentum
6:4 2SGD/conv2d_163/kernel/momentum
(:& 2SGD/conv2d_163/bias/momentum
6:4 2*SGD/batch_normalization_163/gamma/momentum
5:3 2)SGD/batch_normalization_163/beta/momentum
6:4  2SGD/conv2d_195/kernel/momentum
(:& 2SGD/conv2d_195/bias/momentum
6:4 2*SGD/batch_normalization_195/gamma/momentum
5:3 2)SGD/batch_normalization_195/beta/momentum
7:5 @2SGD/conv2d_1522/kernel/momentum
):'@2SGD/conv2d_1522/bias/momentum
7:5@2+SGD/batch_normalization_1522/gamma/momentum
6:4@2*SGD/batch_normalization_1522/beta/momentum
7:5@@2SGD/conv2d_1556/kernel/momentum
):'@2SGD/conv2d_1556/bias/momentum
7:5@2+SGD/batch_normalization_1556/gamma/momentum
6:4@2*SGD/batch_normalization_1556/beta/momentum
?2?
+__inference_model_314_layer_call_fn_7475287
+__inference_model_314_layer_call_fn_7476483
+__inference_model_314_layer_call_fn_7475588
+__inference_model_314_layer_call_fn_7476350?
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
F__inference_model_314_layer_call_and_return_conditional_losses_7474985
F__inference_model_314_layer_call_and_return_conditional_losses_7474817
F__inference_model_314_layer_call_and_return_conditional_losses_7475983
F__inference_model_314_layer_call_and_return_conditional_losses_7476217?
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
"__inference__wrapped_model_7472387?
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
?2?
*__inference_conv2d_6_layer_call_fn_7476502?
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_7476493?
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
7__inference_batch_normalization_6_layer_call_fn_7476566
7__inference_batch_normalization_6_layer_call_fn_7476553
7__inference_batch_normalization_6_layer_call_fn_7476617
7__inference_batch_normalization_6_layer_call_fn_7476630?
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476586
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476522
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476540
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476604?
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
)__inference_re_lu_6_layer_call_fn_7476640?
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_7476635?
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
1__inference_max_pooling2d_6_layer_call_fn_7472503?
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
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_7472497?
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
+__inference_conv2d_27_layer_call_fn_7476659?
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_7476650?
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
8__inference_batch_normalization_27_layer_call_fn_7476723
8__inference_batch_normalization_27_layer_call_fn_7476710
8__inference_batch_normalization_27_layer_call_fn_7476787
8__inference_batch_normalization_27_layer_call_fn_7476774?
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
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476743
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476761
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476697
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476679?
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
*__inference_re_lu_27_layer_call_fn_7476797?
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
E__inference_re_lu_27_layer_call_and_return_conditional_losses_7476792?
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
+__inference_conv2d_35_layer_call_fn_7476816?
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_7476807?
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
8__inference_batch_normalization_35_layer_call_fn_7476931
8__inference_batch_normalization_35_layer_call_fn_7476944
8__inference_batch_normalization_35_layer_call_fn_7476880
8__inference_batch_normalization_35_layer_call_fn_7476867?
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476854
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476836
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476900
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476918?
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
*__inference_re_lu_35_layer_call_fn_7476954?
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_7476949?
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
-__inference_conv2d_1936_layer_call_fn_7476973?
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
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_7476964?
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
:__inference_batch_normalization_1936_layer_call_fn_7477024
:__inference_batch_normalization_1936_layer_call_fn_7477088
:__inference_batch_normalization_1936_layer_call_fn_7477101
:__inference_batch_normalization_1936_layer_call_fn_7477037?
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
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477057
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7476993
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477075
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477011?
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
,__inference_re_lu_1763_layer_call_fn_7477111?
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
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_7477106?
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
-__inference_conv2d_1968_layer_call_fn_7477130?
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
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_7477121?
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
:__inference_batch_normalization_1968_layer_call_fn_7477245
:__inference_batch_normalization_1968_layer_call_fn_7477258
:__inference_batch_normalization_1968_layer_call_fn_7477194
:__inference_batch_normalization_1968_layer_call_fn_7477181?
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
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477150
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477168
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477214
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477232?
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
,__inference_re_lu_1795_layer_call_fn_7477268?
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
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_7477263?
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
:__inference_conv_adjust_channels_235_layer_call_fn_7473216
:__inference_conv_adjust_channels_235_layer_call_fn_7477352
:__inference_conv_adjust_channels_235_layer_call_fn_7477335
:__inference_conv_adjust_channels_235_layer_call_fn_7473181?
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
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477318
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473127
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473145
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477294?
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
)__inference_add_356_layer_call_fn_7477364?
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
D__inference_add_356_layer_call_and_return_conditional_losses_7477358?
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
,__inference_conv2d_163_layer_call_fn_7477383?
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
G__inference_conv2d_163_layer_call_and_return_conditional_losses_7477374?
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
9__inference_batch_normalization_163_layer_call_fn_7477511
9__inference_batch_normalization_163_layer_call_fn_7477434
9__inference_batch_normalization_163_layer_call_fn_7477447
9__inference_batch_normalization_163_layer_call_fn_7477498?
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
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477467
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477421
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477403
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477485?
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
+__inference_re_lu_163_layer_call_fn_7477521?
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
F__inference_re_lu_163_layer_call_and_return_conditional_losses_7477516?
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
,__inference_conv2d_195_layer_call_fn_7477540?
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
G__inference_conv2d_195_layer_call_and_return_conditional_losses_7477531?
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
9__inference_batch_normalization_195_layer_call_fn_7477655
9__inference_batch_normalization_195_layer_call_fn_7477591
9__inference_batch_normalization_195_layer_call_fn_7477668
9__inference_batch_normalization_195_layer_call_fn_7477604?
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
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477560
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477642
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477578
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477624?
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
+__inference_re_lu_195_layer_call_fn_7477678?
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
F__inference_re_lu_195_layer_call_and_return_conditional_losses_7477673?
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
)__inference_add_357_layer_call_fn_7477690?
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
D__inference_add_357_layer_call_and_return_conditional_losses_7477684?
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
-__inference_conv2d_1522_layer_call_fn_7477709?
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
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_7477700?
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
:__inference_batch_normalization_1522_layer_call_fn_7477824
:__inference_batch_normalization_1522_layer_call_fn_7477773
:__inference_batch_normalization_1522_layer_call_fn_7477837
:__inference_batch_normalization_1522_layer_call_fn_7477760?
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
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477811
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477729
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477747
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477793?
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
,__inference_re_lu_1404_layer_call_fn_7477847?
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
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_7477842?
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
4__inference_max_pooling2d_1404_layer_call_fn_7473540?
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
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_7473534?
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
-__inference_conv2d_1556_layer_call_fn_7477866?
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
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_7477857?
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
:__inference_batch_normalization_1556_layer_call_fn_7477917
:__inference_batch_normalization_1556_layer_call_fn_7477981
:__inference_batch_normalization_1556_layer_call_fn_7477994
:__inference_batch_normalization_1556_layer_call_fn_7477930?
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
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477904
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477968
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477886
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477950?
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
,__inference_re_lu_1438_layer_call_fn_7478004?
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
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_7477999?
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
4__inference_max_pooling2d_1438_layer_call_fn_7473656?
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
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_7473650?
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
-__inference_flatten_315_layer_call_fn_7478015?
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
H__inference_flatten_315_layer_call_and_return_conditional_losses_7478010?
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
+__inference_dense_630_layer_call_fn_7478035?
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
F__inference_dense_630_layer_call_and_return_conditional_losses_7478026?
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
+__inference_dense_631_layer_call_fn_7478055?
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
F__inference_dense_631_layer_call_and_return_conditional_losses_7478046?
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
%__inference_signature_wrapper_7475729input_1"?
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
 
?2?
-__inference_conv2d_2155_layer_call_fn_7478074?
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
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_7478065?
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
:__inference_batch_normalization_2155_layer_call_fn_7478202
:__inference_batch_normalization_2155_layer_call_fn_7478125
:__inference_batch_normalization_2155_layer_call_fn_7478189
:__inference_batch_normalization_2155_layer_call_fn_7478138?
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
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478158
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478176
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478112
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478094?
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
 ?
"__inference__wrapped_model_7472387?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????8?5
.?+
)?&
input_1?????????  
? "5?2
0
	dense_631#? 
	dense_631?????????
?
D__inference_add_356_layer_call_and_return_conditional_losses_7477358?j?g
`?]
[?X
*?'
inputs/0?????????  
*?'
inputs/1?????????  
? "-?*
#? 
0?????????  
? ?
)__inference_add_356_layer_call_fn_7477364?j?g
`?]
[?X
*?'
inputs/0?????????  
*?'
inputs/1?????????  
? " ??????????  ?
D__inference_add_357_layer_call_and_return_conditional_losses_7477684?j?g
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
)__inference_add_357_layer_call_fn_7477690?j?g
`?]
[?X
*?'
inputs/0?????????   
*?'
inputs/1?????????   
? " ??????????   ?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477729?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477747?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477793v????;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
U__inference_batch_normalization_1522_layer_call_and_return_conditional_losses_7477811v????;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
:__inference_batch_normalization_1522_layer_call_fn_7477760?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
:__inference_batch_normalization_1522_layer_call_fn_7477773?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
:__inference_batch_normalization_1522_layer_call_fn_7477824i????;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
:__inference_batch_normalization_1522_layer_call_fn_7477837i????;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477886?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477904?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477950v????;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
U__inference_batch_normalization_1556_layer_call_and_return_conditional_losses_7477968v????;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
:__inference_batch_normalization_1556_layer_call_fn_7477917?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
:__inference_batch_normalization_1556_layer_call_fn_7477930?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
:__inference_batch_normalization_1556_layer_call_fn_7477981i????;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
:__inference_batch_normalization_1556_layer_call_fn_7477994i????;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477403?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477421?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477467v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
T__inference_batch_normalization_163_layer_call_and_return_conditional_losses_7477485v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
9__inference_batch_normalization_163_layer_call_fn_7477434?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_163_layer_call_fn_7477447?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_163_layer_call_fn_7477498i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
9__inference_batch_normalization_163_layer_call_fn_7477511i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7476993rpqrs;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477011rpqrs;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477057?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1936_layer_call_and_return_conditional_losses_7477075?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
:__inference_batch_normalization_1936_layer_call_fn_7477024epqrs;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1936_layer_call_fn_7477037epqrs;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
:__inference_batch_normalization_1936_layer_call_fn_7477088?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1936_layer_call_fn_7477101?pqrsM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477560v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477578v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477624?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_195_layer_call_and_return_conditional_losses_7477642?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
9__inference_batch_normalization_195_layer_call_fn_7477591i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
9__inference_batch_normalization_195_layer_call_fn_7477604i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
9__inference_batch_normalization_195_layer_call_fn_7477655?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_195_layer_call_fn_7477668?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477150?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477168?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477214v????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????   
? ?
U__inference_batch_normalization_1968_layer_call_and_return_conditional_losses_7477232v????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????   
? ?
:__inference_batch_normalization_1968_layer_call_fn_7477181?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1968_layer_call_fn_7477194?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_1968_layer_call_fn_7477245i????;?8
1?.
(?%
inputs?????????   
p
? " ??????????   ?
:__inference_batch_normalization_1968_layer_call_fn_7477258i????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????   ?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478094v????;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478112v????;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478158?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
U__inference_batch_normalization_2155_layer_call_and_return_conditional_losses_7478176?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
:__inference_batch_normalization_2155_layer_call_fn_7478125i????;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
:__inference_batch_normalization_2155_layer_call_fn_7478138i????;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
:__inference_batch_normalization_2155_layer_call_fn_7478189?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
:__inference_batch_normalization_2155_layer_call_fn_7478202?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476679rJKLM;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476697rJKLM;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476743?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_7476761?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_27_layer_call_fn_7476710eJKLM;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
8__inference_batch_normalization_27_layer_call_fn_7476723eJKLM;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_27_layer_call_fn_7476774?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_27_layer_call_fn_7476787?JKLMM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476836?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476854?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476900r]^_`;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_7476918r]^_`;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
8__inference_batch_normalization_35_layer_call_fn_7476867?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_35_layer_call_fn_7476880?]^_`M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_35_layer_call_fn_7476931e]^_`;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
8__inference_batch_normalization_35_layer_call_fn_7476944e]^_`;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476522?3456M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476540?3456M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476586r3456;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7476604r3456;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
7__inference_batch_normalization_6_layer_call_fn_7476553?3456M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_6_layer_call_fn_7476566?3456M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_6_layer_call_fn_7476617e3456;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
7__inference_batch_normalization_6_layer_call_fn_7476630e3456;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
H__inference_conv2d_1522_layer_call_and_return_conditional_losses_7477700n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????  @
? ?
-__inference_conv2d_1522_layer_call_fn_7477709a??7?4
-?*
(?%
inputs?????????   
? " ??????????  @?
H__inference_conv2d_1556_layer_call_and_return_conditional_losses_7477857n??7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
-__inference_conv2d_1556_layer_call_fn_7477866a??7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
G__inference_conv2d_163_layer_call_and_return_conditional_losses_7477374n??7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
,__inference_conv2d_163_layer_call_fn_7477383a??7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
H__inference_conv2d_1936_layer_call_and_return_conditional_losses_7476964lij7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1936_layer_call_fn_7476973_ij7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
G__inference_conv2d_195_layer_call_and_return_conditional_losses_7477531n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_conv2d_195_layer_call_fn_7477540a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
H__inference_conv2d_1968_layer_call_and_return_conditional_losses_7477121l|}7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
-__inference_conv2d_1968_layer_call_fn_7477130_|}7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
H__inference_conv2d_2155_layer_call_and_return_conditional_losses_7478065n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????  
? ?
-__inference_conv2d_2155_layer_call_fn_7478074a??7?4
-?*
(?%
inputs?????????   
? " ??????????  ?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_7476650lCD7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_27_layer_call_fn_7476659_CD7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
F__inference_conv2d_35_layer_call_and_return_conditional_losses_7476807lVW7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_35_layer_call_fn_7476816_VW7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_7476493l,-7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_6_layer_call_fn_7476502_,-7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473127{??????<?9
2?/
)?&
input_1?????????   
p
? "-?*
#? 
0?????????  
? ?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7473145{??????<?9
2?/
)?&
input_1?????????   
p 
? "-?*
#? 
0?????????  
? ?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477294z??????;?8
1?.
(?%
inputs?????????   
p
? "-?*
#? 
0?????????  
? ?
U__inference_conv_adjust_channels_235_layer_call_and_return_conditional_losses_7477318z??????;?8
1?.
(?%
inputs?????????   
p 
? "-?*
#? 
0?????????  
? ?
:__inference_conv_adjust_channels_235_layer_call_fn_7473181n??????<?9
2?/
)?&
input_1?????????   
p
? " ??????????  ?
:__inference_conv_adjust_channels_235_layer_call_fn_7473216n??????<?9
2?/
)?&
input_1?????????   
p 
? " ??????????  ?
:__inference_conv_adjust_channels_235_layer_call_fn_7477335m??????;?8
1?.
(?%
inputs?????????   
p
? " ??????????  ?
:__inference_conv_adjust_channels_235_layer_call_fn_7477352m??????;?8
1?.
(?%
inputs?????????   
p 
? " ??????????  ?
F__inference_dense_630_layer_call_and_return_conditional_losses_7478026a??1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? ?
+__inference_dense_630_layer_call_fn_7478035T??1?.
'?$
"?
inputs???????????
? "????????????
F__inference_dense_631_layer_call_and_return_conditional_losses_7478046_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
+__inference_dense_631_layer_call_fn_7478055R??0?-
&?#
!?
inputs??????????
? "??????????
?
H__inference_flatten_315_layer_call_and_return_conditional_losses_7478010b7?4
-?*
(?%
inputs?????????  @
? "'?$
?
0???????????
? ?
-__inference_flatten_315_layer_call_fn_7478015U7?4
-?*
(?%
inputs?????????  @
? "?????????????
O__inference_max_pooling2d_1404_layer_call_and_return_conditional_losses_7473534?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1404_layer_call_fn_7473540?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_max_pooling2d_1438_layer_call_and_return_conditional_losses_7473650?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_max_pooling2d_1438_layer_call_fn_7473656?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_7472497?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_6_layer_call_fn_7472503?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_model_314_layer_call_and_return_conditional_losses_7474817?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????@?=
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
F__inference_model_314_layer_call_and_return_conditional_losses_7474985?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????@?=
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
F__inference_model_314_layer_call_and_return_conditional_losses_7475983?f,-3456CDJKLMVW]^_`ijpqrs|}????????????????????????????????????????<
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
F__inference_model_314_layer_call_and_return_conditional_losses_7476217?f,-3456CDJKLMVW]^_`ijpqrs|}????????????????????????????????????????<
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
+__inference_model_314_layer_call_fn_7475287?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????@?=
6?3
)?&
input_1?????????  
p

 
? "??????????
?
+__inference_model_314_layer_call_fn_7475588?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????@?=
6?3
)?&
input_1?????????  
p 

 
? "??????????
?
+__inference_model_314_layer_call_fn_7476350?f,-3456CDJKLMVW]^_`ijpqrs|}????????????????????????????????????????<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
+__inference_model_314_layer_call_fn_7476483?f,-3456CDJKLMVW]^_`ijpqrs|}????????????????????????????????????????<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
G__inference_re_lu_1404_layer_call_and_return_conditional_losses_7477842h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
,__inference_re_lu_1404_layer_call_fn_7477847[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
G__inference_re_lu_1438_layer_call_and_return_conditional_losses_7477999h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
,__inference_re_lu_1438_layer_call_fn_7478004[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
F__inference_re_lu_163_layer_call_and_return_conditional_losses_7477516h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
+__inference_re_lu_163_layer_call_fn_7477521[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1763_layer_call_and_return_conditional_losses_7477106h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1763_layer_call_fn_7477111[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
G__inference_re_lu_1795_layer_call_and_return_conditional_losses_7477263h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
,__inference_re_lu_1795_layer_call_fn_7477268[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
F__inference_re_lu_195_layer_call_and_return_conditional_losses_7477673h7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
+__inference_re_lu_195_layer_call_fn_7477678[7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
E__inference_re_lu_27_layer_call_and_return_conditional_losses_7476792h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_re_lu_27_layer_call_fn_7476797[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_re_lu_35_layer_call_and_return_conditional_losses_7476949h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_re_lu_35_layer_call_fn_7476954[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
D__inference_re_lu_6_layer_call_and_return_conditional_losses_7476635h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
)__inference_re_lu_6_layer_call_fn_7476640[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
%__inference_signature_wrapper_7475729?f,-3456CDJKLMVW]^_`ijpqrs|}??????????????????????????????????????C?@
? 
9?6
4
input_1)?&
input_1?????????  "5?2
0
	dense_631#? 
	dense_631?????????
