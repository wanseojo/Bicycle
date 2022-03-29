#!/usr/bin/env python
import sys, os
import argparse
import uuid
import numpy as np
import google.protobuf
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import cPickle

def parse_args():
    parser = argparse.ArgumentParser(description='caffe2svnet3')
    parser.register('type', bool, (lambda x: x.lower() in ("yes", "y", "true", "t", "1")))
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--def_int8', dest='prototxt_int8',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net_int8', dest='caffemodel_int8',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--quantize_info', dest='quantize_info',
                        help='quantize info',
                        default=None, type=str)
    parser.add_argument('--out', dest='out',
                        help='model to save',
                        default=None, type=str)
    parser.add_argument('--name', dest='name',
                        help='name of the model',
                        default=None, type=str)
    parser.add_argument('--header', dest='header',
                        help='convert to header file',
                        default=True, type=bool)
    parser.add_argument('--skip_weights', dest='skip_weights',
                        help='skip to write weights files',
                        default=False, type=bool)
    parser.add_argument('--odn', dest='odn',
                        help='0 = skip to make CFNet, RPNet, BRNet',
                        default=False, type=bool)
    parser.add_argument('--adn', dest='adn',
                        help='0 =skip to make ADNet',
                        default=False, type=bool)
    parser.add_argument('--mdn', dest='mdn',
                        help='0 =skip to make MDNet',
                        default=False, type=bool)
    parser.add_argument('--region', dest='region',
                        help='Nation option',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def parse_param_int(strparam):
    ret = []
    spstr1 = strparam.split("]")
    for str1 in spstr1:
        if str1.strip() != "":
            spstr2 = str1.split(",")
            ret2 = []
            for str2 in spstr2:
                str2 = str2.strip()
                str2 = str2.replace("[", "")
                str2 = str2.replace("]", "")
                if str2 != "":
                    ret2.append(float(str(str2)))
            ret.append(ret2)
    return ret

def parse_param_str(strparam):
    spstr1 = strparam.split(",")
    ret = []
    for str1 in spstr1:
        str1 = str1.strip()
        str1 = str1.replace("[", "")
        str1 = str1.replace("]", "")
        if str1 != "":
            ret.append(str(str1))
    return ret

def is_roidata(layers, idx):
    return layers[idx].type == "ROIData"

def is_roipooling3x3way_multi_out(layers, idx):
    return layers[idx].type == "ROIPooling" and layers[idx + 1].type == "ROIPooling" and layers[idx + 2].type == "ROIPooling" and layers[idx + 3].type == "Concat" and \
        layers[idx + 4].type == "ROIPooling" and layers[idx + 5].type == "ROIPooling" and layers[idx + 6].type == "ROIPooling" and layers[idx + 7].type == "Concat" and \
        layers[idx + 8].type == "ROIPooling" and layers[idx + 9].type == "ROIPooling" and layers[idx + 10].type == "ROIPooling" and layers[idx + 11].type == "Concat" and \
        layers[idx + 12].type == "ProposalSlice" and layers[idx + 13].type == "ProposalSlice" and layers[idx + 14].type == "ProposalSlice"

def is_roipooling3x3way(layers, idx):
    return layers[idx].type == "ROIPooling" and layers[idx + 1].type == "ROIPooling" and layers[idx + 2].type == "ROIPooling" and layers[idx + 3].type == "Concat" and \
        layers[idx + 4].type == "ROIPooling" and layers[idx + 5].type == "ROIPooling" and layers[idx + 6].type == "ROIPooling" and layers[idx + 7].type == "Concat" and \
        layers[idx + 8].type == "ROIPooling" and layers[idx + 9].type == "ROIPooling" and layers[idx + 10].type == "ROIPooling" and layers[idx + 11].type == "Concat" and \
        not is_roipooling3x3way_multi_out(layers, idx)

def is_roipooling3way(layers, idx):
    return layers[idx].type == "ROIPooling" and layers[idx + 1].type == "ROIPooling" and layers[idx + 2].type == "ROIPooling" and layers[idx + 3].type == "Concat" and \
        not is_roipooling3x3way(layers, idx - 8) and not is_roipooling3x3way(layers, idx - 4) and not is_roipooling3x3way(layers, idx) and \
        not is_roipooling3x3way_multi_out(layers, idx - 8) and not is_roipooling3x3way_multi_out(layers, idx - 4) and not is_roipooling3x3way_multi_out(layers, idx)

def is_roipooling(layers, idx):
    return layers[idx].type == "ROIPooling" and not is_roipooling3way(layers, idx - 2) and not is_roipooling3way(layers, idx - 1) and not is_roipooling3way(layers, idx) and \
            not is_roipooling3x3way(layers, idx - 10) and not is_roipooling3x3way(layers, idx - 9) and not is_roipooling3x3way(layers, idx - 8) and \
            not is_roipooling3x3way(layers, idx - 6) and not is_roipooling3x3way(layers, idx - 5) and not is_roipooling3x3way(layers, idx - 4) and \
            not is_roipooling3x3way(layers, idx - 2) and not is_roipooling3x3way(layers, idx - 1) and not is_roipooling3x3way(layers, idx) and \
            not is_roipooling3x3way_multi_out(layers, idx - 10) and not is_roipooling3x3way_multi_out(layers, idx - 9) and not is_roipooling3x3way_multi_out(layers, idx - 8) and \
            not is_roipooling3x3way_multi_out(layers, idx - 6) and not is_roipooling3x3way_multi_out(layers, idx - 5) and not is_roipooling3x3way_multi_out(layers, idx - 4) and \
            not is_roipooling3x3way_multi_out(layers, idx - 2) and not is_roipooling3x3way_multi_out(layers, idx - 1) and not is_roipooling3x3way_multi_out(layers, idx)

def is_psroipooling3way(layers, idx):
    return layers[idx].type == "PSROIPooling" and layers[idx + 1].type == "Pooling" and layers[idx + 2].type == "Reshape" and layers[idx + 3].type == "PSROIPooling" and layers[idx + 4].type == "Pooling" and layers[idx + 5].type == "Reshape" and \
        layers[idx + 6].type == "PSROIPooling" and layers[idx + 7].type == "Pooling" and layers[idx + 8].type == "Reshape" and layers[idx + 9].type == "PSROIPooling" and layers[idx + 10].type == "Pooling" and layers[idx + 11].type == "Reshape" and \
        layers[idx + 12].type == "PSROIPooling" and layers[idx + 13].type == "Pooling" and layers[idx + 14].type == "Reshape" and layers[idx + 15].type == "PSROIPooling" and layers[idx + 16].type == "Pooling" and layers[idx + 17].type == "Reshape" and \
        layers[idx + 18].type == "Eltwise" and layers[idx + 19].type == "Eltwise"

def is_psroipooling(layers, idx):
    return layers[idx].type == "PSROIPooling" and layers[idx + 1].type == "Pooling" and layers[idx + 2].type == "Reshape" and layers[idx + 3].type == "PSROIPooling" and layers[idx + 4].type == "Pooling" and layers[idx + 5].type == "Reshape" and \
        not is_psroipooling3way(layers, idx - 15) and not is_psroipooling3way(layers, idx - 12) and not is_psroipooling3way(layers, idx - 9) and not is_psroipooling3way(layers, idx - 6) and not is_psroipooling3way(layers, idx  - 3) and not is_psroipooling3way(layers, idx)

def is_conv_layer(layers, idx):
    return (layers[idx].type == "Convolution" or layers[idx].type == "ConvolutionRistretto" or layers[idx].type == "FusedConvBatchNorm" or layers[idx].type == "FusedConvBatchNormRistretto" or layers[idx].type == "ConvolutionDepthwise")

def is_conv(layers, idx):
    return is_conv_layer(layers, idx) and not is_conv_relu(layers, idx) and not is_conv_eltwise_relu(layers, idx)

def is_conv_relu(layers, idx):
    return is_conv_layer(layers, idx) and (layers[idx + 1].type == "ReLU" or (is_batchnorm(layers, idx + 1) and ((layers[idx + 2].type == "Scale" and layers[idx + 3].type == "ReLU") or (layers[idx + 2].type == "ReLU"))))

def is_conv_erelu1(layers, idx):
    return is_conv_layer(layers, idx) and layers[idx + 1].type == "EReLU"

def is_conv_erelu2(layers, idx):
    return is_conv_layer(layers, idx) and is_batchnorm(layers, idx + 1) and layers[idx + 2].type == "EReLU"

def is_conv_erelu3(layers, idx):
    return is_conv_layer(layers, idx) and is_batchnorm(layers, idx + 1) and layers[idx + 2].type == "Scale" and layers[idx + 3].type == "EReLU"

def is_conv_eltwise_relu2(layers, idx):
    return is_conv_layer(layers, idx) and layers[idx + 1].type == "Eltwise" and layers[idx + 2].type == "ReLU"

def is_conv_eltwise_relu3(layers, idx):
    return is_conv_layer(layers, idx) and is_batchnorm(layers, idx + 1) and layers[idx + 2].type == "Eltwise" and layers[idx + 3].type == "ReLU"

def is_conv_eltwise_relu4(layers, idx):
    return is_conv_layer(layers, idx) and is_batchnorm(layers, idx + 1) and layers[idx + 2].type == "Scale" and layers[idx + 3].type == "Eltwise" and layers[idx + 4].type == "ReLU"

def is_conv_eltwise_relu(layers, idx):
    return is_conv_erelu1(layers, idx) or is_conv_erelu2(layers, idx) or is_conv_erelu3(layers, idx) or is_conv_eltwise_relu2(layers, idx) or is_conv_eltwise_relu3(layers, idx) or is_conv_eltwise_relu4(layers, idx)

def is_deconv(layers, idx):
    return layers[idx].type == "Deconvolution" or layers[idx].type == "DeconvolutionRistretto"

def is_fc(layers, idx):
    return layers[idx].type == "InnerProduct" or layers[idx].type == "FcRistretto" or layers[idx].type == "BinaryInnerProduct" or layers[idx].type == "FusedBinaryFcBatchNorm" or layers[idx].type == "FusedBinaryFcBatchNormRistretto"

def is_fused(layers, idx):
    return layers[idx].type == "FusedConvBatchNorm" or layers[idx].type == "FusedConvBatchNormRistretto" or layers[idx].type == "FusedBinaryFcBatchNorm" or layers[idx].type == "FusedBinaryFcBatchNormRistretto"

def is_binary(layers, idx):
    return layers[idx].type == "BinaryInnerProduct" or layers[idx].type == "FusedBinaryFcBatchNorm" or layers[idx].type == "FusedBinaryFcBatchNormRistretto"

def is_batchnorm(layers, idx):
    return layers[idx].type == "BatchNorm" and ((len(layers[idx].blobs) == 3 and layers[idx + 1].type == "Scale") or (len(layers[idx].blobs) >= 3 and layers[idx + 1].type != "Scale"))

def is_crelu(layers, idx):
    return (layers[idx].type == "Scale" and layers[idx + 1].type == "Concat" and layers[idx + 2].type == "ReLU") or layers[idx].type == "CReLU" 

def is_scale(layers, idx):
    return layers[idx].type == "Scale" and not is_batchnorm(layers, idx - 1) and not is_crelu(layers, idx) and not is_laplacian(layers, idx)

def is_proposalnway(layers, idx):
    return layers[idx].type == "ProposalNway"
    
def is_quantize(layers, idx):
    return layers[idx].type == "Quantize"

def is_eltwise(layers, idx):
    return layers[idx].type == "Eltwise" and \
            not is_conv_erelu1(layers, idx - 1) and not is_conv_erelu2(layers, idx - 2) and not is_conv_erelu3(layers, idx - 3) and \
            not is_conv_eltwise_relu2(layers, idx - 1) and not is_conv_eltwise_relu3(layers, idx - 2) and not is_conv_eltwise_relu4(layers, idx - 3) and \
            not is_laplacian(layers, idx - 3) and not is_laplacian(layers, idx - 4) and not is_laplacian(layers, idx - 5) and not is_psroipooling3way(layers, idx - 18) and not is_psroipooling3way(layers, idx - 19)

def is_relu(layers, idx):
    return layers[idx].type == "ReLU" and not is_conv_relu(layers, idx - 1) and not is_conv_relu(layers, idx - 2) and not is_conv_eltwise_relu2(layers, idx - 2) and not is_crelu(layers, idx - 2) and not is_conv_relu(layers, idx - 3) and not is_conv_eltwise_relu3(layers, idx - 3) and not is_conv_eltwise_relu4(layers, idx - 4)

def is_erelu(layers, idx):
    return layers[idx].type == "EReLU" and not is_conv_erelu1(layers, idx - 1) and not is_conv_erelu2(layers, idx - 2) and not is_conv_erelu3(layers, idx - 3)

def is_elu(layers, idx):
    return layers[idx].type == "ELU"

def is_mish(layers, idx):
    return layers[idx].type == "Mish"

def is_pooling(layers, idx):
    return layers[idx].type == "Pooling" and not is_psroipooling3way(layers, idx - 1) and not is_laplacian(layers, idx - 1) and not is_laplacian(layers, idx - 2) and not is_psroipooling3way(layers, idx - 4) and not is_psroipooling3way(layers, idx - 7) and not is_psroipooling3way(layers, idx - 10) and not is_psroipooling3way(layers, idx - 13) and not is_psroipooling3way(layers, idx - 16) and not is_psroipooling(layers, idx - 1) and not is_psroipooling(layers, idx - 4)

def is_stixelpooling(layers, idx):
    return layers[idx].type == "StixelPooling"

def is_concat(layers, idx):
    return layers[idx].type == "Concat" and not is_crelu(layers, idx - 1) and not is_roipooling3way(layers, idx - 3) and \
        not is_roipooling3x3way(layers, idx - 3) and not is_roipooling3x3way(layers, idx - 7) and not is_roipooling3x3way(layers, idx - 11) and \
        not is_roipooling3x3way_multi_out(layers, idx - 3) and not is_roipooling3x3way_multi_out(layers, idx - 7) and not is_roipooling3x3way_multi_out(layers, idx - 11)

def is_slice(layers, idx):
    return layers[idx].type == "Slice"
def is_sliceconcat(layers, idx):
    return layers[idx].type == "SliceConcat"
def is_reshape(layers, idx):
    return layers[idx].type == "Reshape" and not is_psroipooling3way(layers, idx - 2) and not is_psroipooling3way(layers, idx - 5) and not is_psroipooling3way(layers, idx - 8) and not is_psroipooling3way(layers, idx - 11) and not is_psroipooling3way(layers, idx - 14) and not is_psroipooling3way(layers, idx - 17) and not is_psroipooling(layers, idx - 2) and not is_psroipooling(layers, idx - 5)

def is_softmax(layers, idx):
    return layers[idx].type == "Softmax"

def is_sigmoid(layers, idx):
    return layers[idx].type == "Sigmoid"

def is_detection(layers, idx):
    return layers[idx].type == "Detection"

def is_laplacian(layers, idx):
    return layers[idx].type == "Scale" and layers[idx + 1].type == "Pooling" and layers[idx + 2].type == "Pooling" and layers[idx + 3].type == "Eltwise" and layers[idx + 4].type == "Eltwise" and layers[idx + 5].type == "Eltwise"

def is_segmentation(layers, idx):
    return layers[idx].type == "Segmentation"

def is_curvefitting(layers, idx):
    return layers[idx].type == "CurveFitting"

def is_detection_3d(layers, idx):
    return layers[idx].type == "Detection3D"
    
def is_post_detection(layers, idx):
    return layers[idx].type == "PostDetection"

def is_space2depth(layers, idx):
    return layers[idx].type == "Space2Depth"

def is_depth2space(layers, idx):
    return layers[idx].type == "Depth2Space"

def is_quantized_mask_pooling(layers, idx):
    return layers[idx].type == "QuantizedMaskPooling"

def is_upsample(layers, idx):
    return layers[idx].type == "Upsample"

def is_split(layers, idx):
    return layers[idx].type == "Split"

def is_grid_conv_start(layers, idx):
    return is_slice(layers, idx) and is_slice(layers, idx + 1) and is_slice(layers, idx + 2) and is_slice(layers, idx + 3) and is_slice(layers, idx + 4) \
        and is_conv(layers, idx + 5) and is_conv(layers, idx + 6) and is_conv(layers, idx + 7) and is_conv(layers, idx + 8) \
        and is_conv(layers, idx + 9) and is_conv(layers, idx + 10) and is_conv(layers, idx + 11) and is_conv(layers, idx + 12) \
        and is_conv(layers, idx + 13) and is_conv(layers, idx + 14) and is_conv(layers, idx + 15) and is_conv(layers, idx + 16) \
        and is_conv(layers, idx + 17) and is_conv(layers, idx + 18) and is_conv(layers, idx + 19) and is_conv(layers, idx + 20) \
        and is_concat(layers, idx + 21) and is_concat(layers, idx + 22) and is_concat(layers, idx + 23) and is_concat(layers, idx + 24) and is_concat(layers, idx + 25)

def is_grid_conv_end(layers, idx):
    return is_grid_conv_start(layers, idx - 25)

def get_layer_def(net, proto, idx):
    for layer_def in proto.layer:
        if net._layer_names[idx] == layer_def.name and (len(layer_def.include) == 0 or layer_def.include[0].phase == caffe.TEST):
            return layer_def
    return None

def get_layer_def_by_name(proto, name):
    for layer_def in proto.layer:
        if name == layer_def.name and (len(layer_def.include) == 0 or layer_def.include[0].phase == caffe.TEST):
            return layer_def
    return None

def get_top(layer_def, blob_list_index, idx):
    top = "data_blobs[{}]".format(blob_list_index[layer_def.top[idx]])
    return top
        
def get_bottom(layer_def, blob_list_index, idx):
    bottom = "data_blobs[{}]".format(blob_list_index[layer_def.bottom[idx]])
    return bottom

def is_fold(layer_def, proto, layers_fold):
    for layer_fold in layers_fold:
        if layer_def.name[:len(layer_fold)] == layer_fold:
            layer_reshape = layer_fold + "_reshape" + layer_def.name[len(layer_fold):]
            reshape_layer_def = get_layer_def_by_name(proto, layer_reshape)
            if (reshape_layer_def is None):
                layer_reshape = layer_fold + "_reshape"
                reshape_layer_def = get_layer_def_by_name(proto, layer_reshape)
            layer_d2s = layer_fold + "_d2s" + layer_def.name[len(layer_fold):]
            d2s_layer_def = get_layer_def_by_name(proto, layer_d2s)
            if (d2s_layer_def is None):
                layer_d2s = layer_fold + "_d2s"
                d2s_layer_def = get_layer_def_by_name(proto, layer_d2s)
            if (layer_def.type == "Convolution" and layer_def.convolution_param.num_output == 2) or \
                (layer_def.type == "InnerProduct" and layer_def.inner_product_param.num_output == 2) or \
                (reshape_layer_def is not None and reshape_layer_def.reshape_param.shape.dim[1] == 2) or \
                (d2s_layer_def is not None and  layer_def.convolution_param.num_output / (d2s_layer_def.depth2space_param.scale * d2s_layer_def.depth2space_param.scale) == 2):
                return True
    return False

def is_disable_layer(layer_def, net, disable_layers_for_speedup_rpn):
    return (layer_def.name in disable_layers_for_speedup_rpn or layer_def.name[:-1] in disable_layers_for_speedup_rpn) and \
        ((layer_def.type != "Softmax" and layer_def.type != "Reshape") or \
        (layer_def.type == "Softmax" and net.blobs[layer_def.bottom[0]].shape[1] == 2) or \
        (layer_def.type == "Reshape" and ("score" in layer_def.name and net.blobs[layer_def.top[0]].shape[1] == 2) or ("prob" in layer_def.name and net.blobs[layer_def.bottom[0]].shape[1] == 2)))

def is_deconv_resize(net, proto, idx):
    if net.layers[idx].type == "Deconvolution" or net.layers[idx].type == "DeconvolutionRistretto":
        layer_def = get_layer_def(net, proto, idx)
        if layer_def.convolution_param.stride[0] == 2 \
            and layer_def.convolution_param.kernel_size[0] == 2 \
            and layer_def.convolution_param.weight_filler.type == "constant" \
            and layer_def.convolution_param.bias_term == False \
            and net.blobs[layer_def.bottom[0]].channels == net.blobs[layer_def.top[0]].channels \
            and layer_def.convolution_param.group == net.blobs[layer_def.top[0]].channels:
            return True
        else:
            return False
    else:
        return False

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums["reverse_mapping"] = reverse
    return type("Enum", (), enums)

FUNC_TYPE = enum("ComputeDeepConv", "Detect_OD", "Detect_AD", "Detect_MD", "Detect_VL", "Detect_ES", "Detect_CL", "Detect_FS", "Recognize_TSR", "Recognize_TLR", "Recognize_FPR", "Segment_FSD", "Segment_LSD", "Segment_LD", "Segment_LF", "Segment_LBF", "Segment_LB", "CFNetProc", "RPNetProc", "BRNetProc", "ADNetProc", "MDNetProc")

func_names = {}
func_names[FUNC_TYPE.ComputeDeepConv] = "ComputeDeepConv"
func_names[FUNC_TYPE.Detect_OD] = "Detect"
func_names[FUNC_TYPE.Detect_AD] = "Detect"
func_names[FUNC_TYPE.Detect_MD] = "Detect"
func_names[FUNC_TYPE.Detect_VL] = "Detect"
func_names[FUNC_TYPE.Detect_ES] = "Detect"
func_names[FUNC_TYPE.Detect_CL] = "Detect"
func_names[FUNC_TYPE.Detect_FS] = "Detect"
func_names[FUNC_TYPE.Recognize_TSR] = "ComputeTSR"
func_names[FUNC_TYPE.Recognize_TLR] = "ComputeTLR"
func_names[FUNC_TYPE.Recognize_FPR] = "ComputeFPR"
func_names[FUNC_TYPE.Segment_FSD] = "Segment"
func_names[FUNC_TYPE.Segment_LSD] = "Segment"
func_names[FUNC_TYPE.Segment_LD] = "Segment"
func_names[FUNC_TYPE.Segment_LF] = "Segment"
func_names[FUNC_TYPE.Segment_LB] = "Segment"
func_names[FUNC_TYPE.Segment_LBF] = "Segment"
func_names[FUNC_TYPE.CFNetProc] = "Process"
func_names[FUNC_TYPE.RPNetProc] = "Process"
func_names[FUNC_TYPE.BRNetProc] = "Process"
func_names[FUNC_TYPE.ADNetProc] = "Process"
func_names[FUNC_TYPE.MDNetProc] = "Process"

functions = {}
functions[FUNC_TYPE.ComputeDeepConv] = "ComputeDeepConv(const Image& image, Tensor& rois)"
functions[FUNC_TYPE.Detect_OD] = "Detect(const Tensor& rois, std::vector<Object>& objects, bool restore_scale, int refine_size)"
functions[FUNC_TYPE.Detect_AD] = "Detect(const std::unordered_map<std::string, const Tensor*>& feat_maps, const Tensor& rois, std::vector<Attribute>& attributes, bool restore_scale)"
functions[FUNC_TYPE.Detect_MD] = "Detect(const std::unordered_map<std::string, const Tensor*>& feat_maps, const Tensor& rois, std::vector<Mask>& masks)"
functions[FUNC_TYPE.Detect_VL] = "Detect(Tensor& VL)"
functions[FUNC_TYPE.Detect_ES] = "Detect(const Tensor& ego_rois, Tensor& Ego)"
functions[FUNC_TYPE.Detect_CL] = "Detect(const Tensor& closeness_rois, Tensor& Close)"
functions[FUNC_TYPE.Detect_FS] = "Detect(const Tensor& fs_rois, Tensor& FS)"
functions[FUNC_TYPE.Recognize_TSR] = "ComputeTSR(const Image& image, const FloatRect& tsr_roi)"
functions[FUNC_TYPE.Recognize_TLR] = "ComputeTLR(const Image& image, const FloatRect& tlr_roi)"
functions[FUNC_TYPE.Recognize_FPR] = "ComputeFPR(const std::unordered_map<std::string, const Tensor*>& feat_maps, const Tensor& rois, std::vector<FalsePositive>& false_positives)"
functions[FUNC_TYPE.Segment_FSD] = "Segment(const Image& image, Tensor& seg_map)"
functions[FUNC_TYPE.Segment_LSD] = "Segment(const Image& image, Tensor& seg_map)"
functions[FUNC_TYPE.Segment_LD] = "Segment(const Image& image, std::vector<Tensor*>& ld_maps, std::vector<float_t*>& cluster_minmax_vals)"
functions[FUNC_TYPE.Segment_LF] = "Segment(const Image& image, std::vector<Tensor*>& ld_maps, std::vector<float_t*>& cluster_minmax_vals)"
functions[FUNC_TYPE.Segment_LB] = "Segment(const Image& image, std::vector<Tensor*>& ld_maps, std::vector<float_t*>& cluster_minmax_vals)"
functions[FUNC_TYPE.Segment_LBF] = "Segment(const Image& image, std::vector<Tensor*>& ld_maps, std::vector<float_t*>& cluster_minmax_vals)"
functions[FUNC_TYPE.CFNetProc] = "Process(const Image& image)"
functions[FUNC_TYPE.RPNetProc] = "Process(const IConvFeatureMap* feat_maps, Tensor& proposal_rois)"
functions[FUNC_TYPE.BRNetProc] = "Process(const IConvFeatureMap* feat_maps, const Tensor& rois, Tensor& detection_boxes)"
functions[FUNC_TYPE.ADNetProc] = "Process(const IConvFeatureMap* feat_maps, const Tensor& rois, std::vector<Attribute>& attributes)"
functions[FUNC_TYPE.MDNetProc] = "Process(const IConvFeatureMap* feat_maps, const Tensor& rois, std::vector<Mask>& masks)"

start_layers= {}
start_layers[FUNC_TYPE.ComputeDeepConv] = ["input-data"]
start_layers[FUNC_TYPE.Detect_OD] = ["roi_pool_conv3", "roi_pool_conv2_1x1", "roi_pool_cls_score3", "roi_pool_ncls_score3", "roi_pool_conv1", "ssd_start", "roi_pool_cls_score", "roi_pool"]
start_layers[FUNC_TYPE.Detect_AD] = ["roi_pool_conv3_bbox", "roi_pool_conv2_1x1_3d", "post_roi_pool_conv2_1x1", "post_roi_pool_conv2", "post_roi_pool_conv1", "post_roi_pool"]
start_layers[FUNC_TYPE.Detect_MD] = ["roi_pool_conv2_4x4_seg"]
start_layers[FUNC_TYPE.Detect_VL] = ["conv4_f/pool"]
start_layers[FUNC_TYPE.Detect_ES] = ["ego/conv5/conv"]
start_layers[FUNC_TYPE.Detect_CL] = ["closeness_pool"]
start_layers[FUNC_TYPE.Detect_FS] = ["fs_roi_pool_conv4_f"]
start_layers[FUNC_TYPE.Recognize_TSR] = ["input-data"]
start_layers[FUNC_TYPE.Recognize_TLR] = ["input-data"]
start_layers[FUNC_TYPE.Recognize_FPR] = ["roi_pool_conv1_4x4_fp", "roi_pool_conv2_4x4_fp"]
start_layers[FUNC_TYPE.Segment_FSD] = ["seg-input-data", "fsd-input-data"]
start_layers[FUNC_TYPE.Segment_LSD] = ["seg-input-data", "lsd-input-data"]
start_layers[FUNC_TYPE.Segment_LD] = ["lane-input-data", "zf_lane-input-data"]
start_layers[FUNC_TYPE.Segment_LF] = ["lane_fsd-input-data"]
start_layers[FUNC_TYPE.Segment_LB] = ["lane_fsd-input-data", "lane_rbd-input-data"]
start_layers[FUNC_TYPE.Segment_LBF] = ["lane_fsd-input-data", "lane_fsd_rbd-input-data"]
start_layers[FUNC_TYPE.CFNetProc] = ["input-data"]
start_layers[FUNC_TYPE.RPNetProc] = ["rpn_cls_score2", "af_rpn_cls_score"]
start_layers[FUNC_TYPE.BRNetProc] = ["roi_pool_conv3", "roi_pool_conv2_1x1", "roi_pool_cls_score3", "roi_pool_ncls_score3", "roi_pool_conv1", "ssd_start", "roi_pool_cls_score", "roi_pool"]
start_layers[FUNC_TYPE.ADNetProc] = ["roi_pool_conv3_bbox", "roi_pool_conv2_1x1_3d", "post_roi_pool_conv2_1x1", "post_roi_pool_conv2", "post_roi_pool_conv1", "post_roi_pool"]
start_layers[FUNC_TYPE.MDNetProc] = ["roi_pool_conv2_4x4_seg"]

end_layers = {}
end_layers[FUNC_TYPE.ComputeDeepConv] = ["proposalnway"]
end_layers[FUNC_TYPE.Detect_OD] = ["detection"]
end_layers[FUNC_TYPE.Detect_AD] = ["detection_3d", "post_detection"]
end_layers[FUNC_TYPE.Detect_MD] = ["conv8/out"]
end_layers[FUNC_TYPE.Detect_VL] = ["vpy_out"]
end_layers[FUNC_TYPE.Detect_ES] = ["ego_out"]
end_layers[FUNC_TYPE.Detect_CL] = ["closeness_out"]
end_layers[FUNC_TYPE.Detect_FS] = ["failsafe_out"]
end_layers[FUNC_TYPE.Recognize_TSR] = ["fc_out"]
end_layers[FUNC_TYPE.Recognize_TLR] = ["fc_out"]
end_layers[FUNC_TYPE.Recognize_FPR] = ["fp_prob"]
end_layers[FUNC_TYPE.Segment_FSD] = ["segmentation"]
end_layers[FUNC_TYPE.Segment_LSD] = ["segmentation"]
end_layers[FUNC_TYPE.Segment_LD] = ["clu2typeColor_prob"]
end_layers[FUNC_TYPE.Segment_LF] = ["ld_clu2typeColor_prob"]
end_layers[FUNC_TYPE.Segment_LB] = ["boundary_clu2typeShape_prob"]
end_layers[FUNC_TYPE.Segment_LBF] = ["boundary_clu2typeShape_prob"]
end_layers[FUNC_TYPE.CFNetProc] = ["conv4_f/relu2", "conv_concat_f", "conv_fc7/relu", "conv4_f/crelu"]
end_layers[FUNC_TYPE.RPNetProc] = ["proposalnway"]
end_layers[FUNC_TYPE.BRNetProc] = ["detection"]
end_layers[FUNC_TYPE.ADNetProc] = ["detection_3d", "post_detection"]
end_layers[FUNC_TYPE.MDNetProc] = ["conv8/out"]

class_type_map = dict()
class_type_map["pedestrian"] = "IObjectDetector::ClassType::OD_PEDESTRIAN"
class_type_map["rider"] = "IObjectDetector::ClassType::OD_RIDER"
class_type_map["bicycle"] = "IObjectDetector::ClassType::OD_BICYCLE"
class_type_map["motorbike"] = "IObjectDetector::ClassType::OD_MOTORBIKE"
class_type_map["car"] = "IObjectDetector::ClassType::OD_CAR"
class_type_map["sedan"] = "IObjectDetector::ClassType::OD_SEDAN"
class_type_map["van"] = "IObjectDetector::ClassType::OD_VAN"
class_type_map["truck"] = "IObjectDetector::ClassType::OD_TRUCK"
class_type_map["bus"] = "IObjectDetector::ClassType::OD_BUS"
class_type_map["ts"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN"
class_type_map["ts_c"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_CIRCLE"
class_type_map["ts_t"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_TRIANGLE"
class_type_map["ts_r"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_RECTANGLE"
class_type_map["ts_s"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_RECTANGLE"
class_type_map["tl"] = "IObjectDetector::ClassType::OD_TRAFFIC_LIGHT"
class_type_map["tl_c"] = "IObjectDetector::ClassType::OD_TRAFFIC_LIGHT_CAR"
class_type_map["tl_p"] = "IObjectDetector::ClassType::OD_TRAFFIC_LIGHT_PED"
class_type_map["ts_sup"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_SUP"
class_type_map["ts_supl"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_SUP_LETTER"
class_type_map["ts_supd"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_SUP_DRAWING"
class_type_map["ts_supa"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_SUP_ARROW"
class_type_map["ts_supz"] = "IObjectDetector::ClassType::OD_TRAFFIC_SIGN_SUP_ZONE"

def make_constructor_and_weights(net, proto, net_name, start_layer, end_layer, impl_head, concrete_head, concrete_body, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info, out_path, skip_weights, func=None, use_ifdef=True):
    weights_file_list = []

    def close_wt(fw):
        fw.write("\n};\n")
        fw.write("}; // namespace sv\n")
        if use_ifdef:
            fw.write("#endif\n")
        fw.close()

    def open_wt(fw, net_name, wt_count, suffix=""):
        if fw is not None:
            close_wt(fw)
        try: os.makedirs(os.path.join(out_path, "weights"))
        except OSError: pass
        filename = "{}{}_wt{:03d}.cpp".format(net_name.lower(), suffix, wt_count / 5000000)
        weights_file_list.append(filename)        
        fw = open(os.path.join(out_path, "weights", filename), "w")
        if use_ifdef:
            fw.write("#if defined({}) || defined(SVNETALL)\n".format(net_name.upper()))
        fw.write("namespace sv {\n")
        fw.write("unsigned char {}{}_wt{:03d}[] = {{".format(net_name, suffix, wt_count / 5000000))
        return fw

    def write_wt(fw, net_name, wt_count, wt_stream, suffix=""):
        for i, d in enumerate(wt_stream):
            if wt_count % 1000 == 0:
                fw.write("\n  ")
            fw.write("0x{}, ".format(d.encode("hex")))
            wt_count += 1
            if wt_count % 5000000 == 0:
                fw = open_wt(fw, net_name, wt_count, suffix)
        return fw, wt_count    

    def make_wt_def(net_name, wt_count, suffix=""):
        wt_def = "extern unsigned char "
        for i in xrange(wt_count / 5000000 + 1):
            wt_def += "{}{}_wt{:03d}[], ".format(net_name, suffix, i)
        wt_def = wt_def[:-2] + ";\n"
        wt_def += "const unsigned char* {}{}_wt[] = {{ ".format(net_name, suffix)
        for i in xrange(wt_count / 5000000 + 1):
            wt_def += "{}{}_wt{:03d}, ".format(net_name, suffix, i)
        wt_def = wt_def[:-2] + " };\n\n"
        return wt_def

    fw = None
    fw_int8 = None
    if not skip_weights:
        fw = open_wt(None, net_name, 0)
        if is_int8: fw_int8 = open_wt(None, net_name, 0, "_int8")    
    
    blob_list = []
    layer_list = []

    constructor = ""
    constructor_layer_init = ""
    constructor_binary_fc = ""
    constructor_class_map = ""
    constructor_has_attribute = ""
    constructor_has_mask = ""
    constructor_for_tsr = ""
    constructor_for_tlr = ""
    constructor_checks_fp = ""
    constructor_data_blobs_by_name = ""
    wt_def = ""

    wt_count = 0  
    wt_count_int8 = 0    

    layers_fold = ["rpn_cls_score", "seg_score", "seg-seg_score", "fsd-seg_score", "fsd_seg_score", "lane-seg_score", "lane_seg_score", "vp_seg_score", "ld_seg_score", "boundary_seg_score", "ld_vp_seg_score"]
    start_idx = -1
    end_idx = -1
    for idx, layer in enumerate(net.layers):
        layer_def = get_layer_def(net, proto, idx)
        if layer_def is None:
            continue
        if start_idx == -1 and layer_def.name in start_layer:
            start_idx = idx
        if start_idx == -1:
            continue       

        for bottom in layer_def.bottom:
            if bottom in blob_list:
                continue
            blob_list.append(bottom)
        for top in layer_def.top:
            if top in blob_list:
                continue
            blob_list.append(top)
        layer_idx = len(layer_list)
        layer_list.append(layer_def.name)

        if is_int8:
            for idx_int8, layer_int8 in enumerate(net_int8.layers):
                if layer_def.name == net_int8._layer_names[idx_int8]:
                    break
            layer_def_int8 = get_layer_def(net_int8, proto_int8, idx_int8)

        if is_conv(net.layers, idx) or is_conv_relu(net.layers, idx) or is_conv_eltwise_relu(net.layers, idx) or is_deconv(net.layers, idx) or is_fc(net.layers, idx) or is_scale(net.layers, idx):
            def make_wb(net, idx, layer, proto):           
                w_data = layer.blobs[0].data
                b_data = None

                if len(layer.blobs) > 1:                
                    b_data = layer.blobs[1].data                
                    
                if idx + 1 < len(net.layers) and is_batchnorm(net.layers, idx + 1):
                    if is_deconv(net.layers, idx):
                        w_data_transpose = np.zeros([w_data.shape[1] * layer_def.convolution_param.group, w_data.shape[0] / layer_def.convolution_param.group, w_data.shape[2], w_data.shape[3]])
                        grp_size = w_data.shape[0] / layer_def.convolution_param.group
                        for grp_idx in xrange(layer_def.convolution_param.group):
                            w_data_slice = w_data[grp_idx * grp_size : (grp_idx + 1) * grp_size].transpose(1, 0, 2, 3)
                            w_data_transpose[grp_idx * w_data.shape[1] : (grp_idx + 1) * w_data.shape[1], :, :, :] = w_data_slice[:]

                        w_data = w_data_transpose.astype(np.float32)

                    if b_data is None:
                        b_data = np.zeros(w_data.shape[0])
                        if is_conv(net.layers, idx) or is_conv_relu(net.layers, idx) or is_conv_eltwise_relu(net.layers, idx) or is_deconv(net.layers, idx):
                            layer_def.convolution_param.bias_term = True
                        elif is_fc(net.layers, idx):
                            layer_def.inner_product_param.bias_term = True
                        elif is_scale(net.layers, idx):
                            layer_def.scale_param.bias_term = True

                    bn_layer_def = get_layer_def(net, proto, idx + 1)
                    scale_factor = 0
                    if net.layers[idx + 1].blobs[2].data[0] != 0:
                        scale_factor = 1 / net.layers[idx + 1].blobs[2].data[0]
                    mean = net.layers[idx + 1].blobs[0].data * scale_factor
                    var = (net.layers[idx + 1].blobs[1].data * scale_factor + bn_layer_def.batch_norm_param.eps) ** 0.5                    
                    if len(net.layers[idx + 1].blobs) >= 3 and net.layers[idx + 2].type != "Scale":
                        scale_data = net.layers[idx + 1].blobs[3].data if len(net.layers[idx + 1].blobs) > 3 else np.ones(mean.shape)
                        bias_data = net.layers[idx + 1].blobs[4].data if len(net.layers[idx + 1].blobs) > 4 else np.zeros(mean.shape)                
                        w_data = (w_data.reshape(w_data.shape[0], -1) * scale_data.reshape(scale_data.shape[0], 1) / var.reshape(var.shape[0], 1)).reshape(w_data.shape).astype(np.float32)
                        b_data = ((b_data - mean) * scale_data / var + bias_data).astype(np.float32)
                    elif len(net.layers[idx + 1].blobs) == 3 and net.layers[idx + 2].type == "Scale":
                        w_data = (w_data.reshape(w_data.shape[0], -1) * net.layers[idx + 2].blobs[0].data.reshape(net.layers[idx + 2].blobs[0].data.shape[0], 1) / var.reshape(var.shape[0], 1)).reshape(w_data.shape).astype(np.float32)
                        b_data = ((b_data - mean) * net.layers[idx + 2].blobs[0].data / var + net.layers[idx + 2].blobs[1].data).astype(np.float32)
                    else:
                        raise ValueError("len(batch_norm.blobs) should be 3 or 5, but it is {}".format(len(net.layers[idx + 1].blobs)))

                    if is_deconv(net.layers, idx):
                        w_data_transpose = np.zeros([w_data.shape[1] * layer_def.convolution_param.group, w_data.shape[0] / layer_def.convolution_param.group, w_data.shape[2], w_data.shape[3]])
                        grp_size = w_data.shape[0] / layer_def.convolution_param.group
                        for grp_idx in xrange(layer_def.convolution_param.group):
                            w_data_slice = w_data[grp_idx * grp_size : (grp_idx + 1) * grp_size].transpose(1, 0, 2, 3)
                            w_data_transpose[grp_idx * w_data.shape[1] : (grp_idx + 1) * w_data.shape[1], :, :, :] = w_data_slice[:]

                        w_data = w_data_transpose.astype(np.float32)

                elif is_fused(net.layers, idx):
                    scale_factor = 0
                    if net.layers[idx].blobs[3].data[0] != 0:
                        scale_factor = 1 / net.layers[idx].blobs[3].data[0]
                    mean = net.layers[idx].blobs[1].data * scale_factor
                    var = (net.layers[idx].blobs[2].data * scale_factor + 1e-5) ** 0.5
                    w_data = (w_data.reshape(w_data.shape[0], -1) * net.layers[idx].blobs[4].data.reshape(net.layers[idx].blobs[4].data.shape[0], 1) / var.reshape(var.shape[0], 1)).reshape(w_data.shape).astype(np.float32)
                    b_data = (-mean * net.layers[idx].blobs[4].data / var + net.layers[idx].blobs[5].data).astype(np.float32)

                return w_data, b_data

            w_data, b_data = make_wb(net, idx, layer, proto)

            if not is_deconv_resize(net, proto, idx):
                fold_param = ""
                fold_func = ""
                if is_fold(layer_def, proto, layers_fold):
                    fold_param = ", FOLD"
                    fold_func = "_FOLD"
             
                constructor_layer_init += "  // {}\n".format(layer_def.name)
                if is_binary(net.layers, idx):
                    constructor_layer_init += "  layer_blobs[{}].resize(3, {{}});\n".format(layer_idx)
                    constructor_binary_fc += "  cnn->binarize_weight(layer_blobs[{}][0], layer_blobs[{}][2]);\n".format(layer_idx, layer_idx)
                else:
                    constructor_layer_init += "  layer_blobs[{}].resize(2, {{}});\n".format(layer_idx)
                if len(layer.blobs[0].shape) == 4:
                    constructor_layer_init += "  layer_blobs[{}][0].ReshapeCPU({}, {}, {}, {});\n".format(layer_idx, layer.blobs[0].shape[0], layer.blobs[0].shape[1], layer.blobs[0].shape[2], layer.blobs[0].shape[3])
                elif len(layer.blobs[0].shape) == 3:
                    constructor_layer_init += "  layer_blobs[{}][0].ReshapeCPU({}, {}, {});\n".format(layer_idx, layer.blobs[0].shape[0], layer.blobs[0].shape[1], layer.blobs[0].shape[2])
                elif len(layer.blobs[0].shape) == 2:
                    constructor_layer_init += "  layer_blobs[{}][0].ReshapeCPU({}, {});\n".format(layer_idx, layer.blobs[0].shape[0], layer.blobs[0].shape[1])
                elif len(layer.blobs[0].shape) == 1:
                    constructor_layer_init += "  layer_blobs[{}][0].ReshapeCPU({});\n".format(layer_idx, layer.blobs[0].shape[0])               
                                    
                constructor_layer_init += "  layer_blobs[{}][0].LoadCPU({}_wt, offset{});\n".format(layer_idx, net_name, fold_param)
                if not skip_weights: 
                    fw, wt_count = write_wt(fw, net_name, wt_count, w_data.tobytes())
                else:
                    wt_count += len(w_data.tobytes())

                if b_data is not None:
                    constructor_layer_init += "  layer_blobs[{}][1].ReshapeCPU({});\n".format(layer_idx, b_data.shape[0])
                    constructor_layer_init += "  layer_blobs[{}][1].LoadCPU({}_wt, offset{});\n".format(layer_idx, net_name, fold_param)
                    if not skip_weights: 
                        fw, wt_count = write_wt(fw, net_name, wt_count, b_data.tobytes())
                    else:                            
                        wt_count += len(b_data.tobytes())

                if is_binary(net.layers, idx):
                    constructor_layer_init += "  layer_blobs[{}][2].ReshapeCPU({});\n".format(layer_idx, b_data.shape[0])

                if is_int8 and layer_int8.type.endswith("Ristretto"):                   
                    w_data, b_data = make_wb(net_int8, idx_int8, layer_int8, proto_int8)  

                    constructor_layer_init += "  LOAD_INT_W{}({}, {}, {});\n".format(fold_func, layer_idx, layer_def_int8.quantization_param.fl_params, layer_def_int8.quantization_param.bw_params)
                    if not skip_weights: 
                        fw_int8, wt_count_int8 = write_wt(fw_int8, net_name, wt_count_int8, w_data.tobytes(), "_int8")
                    else:
                        wt_count_int8 += len(w_data.tobytes())
                    if b_data is not None:
                        if layer_def_int8.quantization_param.bw_params_bias == 0:
                            layer_def_int8.quantization_param.bw_params_bias = layer_def_int8.quantization_param.bw_params
                        if layer_def_int8.quantization_param.fl_params_bias == 1000:
                            layer_def_int8.quantization_param.fl_params_bias = layer_def_int8.quantization_param.fl_params
                        constructor_layer_init += "  LOAD_INT_B{}({}, {}, {});\n".format(fold_func, layer_idx, layer_def_int8.quantization_param.fl_params_bias, layer_def_int8.quantization_param.bw_params_bias)
                        if not skip_weights: 
                            fw_int8, wt_count_int8 = write_wt(fw_int8, net_name, wt_count_int8, b_data.tobytes(), "_int8")
                        else:
                            wt_count_int8 += len(b_data.tobytes())
                    else:
                        constructor_layer_init += "  SET_SHIFTN_B({}, {});\n".format(layer_idx, layer_def_int8.quantization_param.fl_params_bias)

                if (is_conv(net.layers, idx) or is_conv_relu(net.layers, idx) or is_conv_eltwise_relu(net.layers, idx)) and layer_def.convolution_param.group > 1:
                    if start_layer != start_layers[FUNC_TYPE.Segment_LD] and start_layer != start_layers[FUNC_TYPE.Segment_LF] and start_layer != start_layers[FUNC_TYPE.Segment_LB] and start_layer != start_layers[FUNC_TYPE.Segment_LBF]:
                        constructor_layer_init += "  group_conv_layers.push_back(std::make_pair({}, {}));\n".format(layer_idx, layer_def.convolution_param.group)

        elif is_proposalnway(net.layers, idx):
            num_rpn = len(layer_def.proposal_nway_param.rpn_option)
            constructor_layer_init += "  // {}\n".format(layer_def.name)
            constructor_layer_init += "  anchors.resize({}, {{}});\n".format(num_rpn)            
            for i in xrange(num_rpn):
                ratios = parse_param_int(layer_def.proposal_nway_param.rpn_option[i].ratios)
                scales = parse_param_int(layer_def.proposal_nway_param.rpn_option[i].scales)
                if len(ratios) > 0 and len(ratios[0]) > 0 and len(scales) > 0 and len(scales[0]) > 0:
                    vec_ratios = "std::vector<float> {"
                    for j in xrange(len(ratios[0])):
                        vec_ratios += "{}, ".format(ratios[0][j])
                    vec_ratios = vec_ratios[:-2] + "}"
                    vec_scales = "std::vector<float> {"
                    for j in xrange(len(scales[0])):
                        vec_scales += "{}, ".format(scales[0][j])
                    vec_scales = vec_scales[:-2] + "}"
                    constructor_layer_init += "  MakeAnchors({}, {}, {}, anchors[{}]);\n".format(layer_def.proposal_nway_param.rpn_option[i].base_size, vec_ratios, vec_scales, i)
                    constructor_layer_init += "#ifdef USE_GPU\n"
                    constructor_layer_init += "  CopyCPUToGPU(anchors[{}]);\n".format(i)
                    constructor_layer_init += "#endif\n"
                else:
                    constructor_layer_init += "  anchors[{}].d1 = 1;\n".format(i)
                    constructor_layer_init += "  anchors[{}].d2 = 4;\n".format(i)
        elif is_detection(net.layers, idx):
            class_names = parse_param_str(proto.class_names)
            constructor_class_map = "\n"
            constructor_class_map += "  const int _class_map[] = { "
            if len(layer_def.detection_param.class_map) == 0 or layer_def.detection_param.class_map == "[]":
                class_map = [i for i in xrange(len(class_names) + 1)]
            else:
                class_map = [ float(cls.strip()) for cls in layer_def.detection_param.class_map.replace("[", "").replace("]", "").split(",") ]
            for i in xrange(len(class_map)):
                constructor_class_map += "{}".format(int(class_map[i]))
                if i < len(class_map) - 1:
                    constructor_class_map += ", "
            constructor_class_map += " };\n"
            constructor_class_map += "  class_map.assign(_class_map, _class_map + {});\n".format(len(class_map))                         
        elif is_detection_3d(net.layers, idx) or is_post_detection(net.layers, idx):      
            if (len(layer_def.bottom) == 4 or len(layer_def.bottom) == 5):
                # 3D car
                attribute_class = ["car", "van", "truck", "excavator", "forklift", "crane", "vehicle", "bus"]                
            else:
                # 3D car + dir + occ + trunc
                attribute_class = ["pedestrian", "rider", "bicycle", "motorbike", "car", "van", "truck", "excavator", "forklift", "crane", "vehicle", "bus"]     
            class_names = parse_param_str(proto.class_names)
            if func == FUNC_TYPE.ADNetProc:        
                constructor_has_attribute = "\n"
                for i in xrange(len(class_names)):
                    has_attribute = False
                    for class_name in attribute_class:
                        if class_name in class_names[i] or class_names[i] in class_name:
                            has_attribute = True
                            break
                    if has_attribute:
                        constructor_has_attribute += "  has_attribute[{}] = true;\n".format(class_type_map[class_names[i]])
                    else:
                        constructor_has_attribute += "  has_attribute[{}] = false;\n".format(class_type_map[class_names[i]])                
                attribute_group = parse_param_str(layer_def.post_detection_param.attribute_group)
                if len(attribute_group) > 0:
                    constructor_has_attribute += "\n"
                    for i in xrange(len(class_names)):
                        constructor_has_attribute += "  attribute_group[{}] = {};\n".format(class_type_map[class_names[i]], attribute_group[i] if i < len(attribute_group) else -1)
                regression_group = parse_param_str(layer_def.post_detection_param.regression_group)
                if len(regression_group) > 0:
                    constructor_has_attribute += "\n"                
                    for i in xrange(len(class_names)):
                        constructor_has_attribute += "  regression_group[{}] = {};\n".format(class_type_map[class_names[i]], regression_group[i] if i < len(regression_group) else -1)
            else:
                constructor_has_attribute += "  const bool _has_attribute[] = { "        
                for i in xrange(len(class_names)):
                    has_attribute = False
                    for class_name in attribute_class:
                        if class_name in class_names[i] or class_names[i] in class_name:
                            has_attribute = True
                            break
                    if has_attribute:
                        constructor_has_attribute += "true"
                    else:
                        constructor_has_attribute += "false"
                    if i < len(class_names) - 1:
                        constructor_has_attribute += ", "
                constructor_has_attribute += " };\n"
                constructor_has_attribute += "  has_attribute.assign(_has_attribute, _has_attribute + {});\n".format(len(class_names))       
            
        if func in [FUNC_TYPE.Recognize_TSR, FUNC_TYPE.Recognize_TLR]:
            # for TSR
            if func == FUNC_TYPE.Recognize_TSR and str(blob_list[-1]) in end_layers[FUNC_TYPE.Recognize_TSR]:
                base_idx = {'C': 1000, 'T': 2000, 'S': 3000, 'P': 4000, 'c': 1000, 't': 2000, 's': 3000, 'p': 4000}
                constructor_for_tsr += "  FC_BLOB_IDX = {};\n".format(len(blob_list) - 1)
                try:
                  constructor_for_tsr += "  BASE_IDX = {};\n".format(base_idx[net_name[6]])
                except KeyError:
                  constructor_for_tsr += "  BASE_IDX = {};\n".format(base_idx[net_name[-1]])
            # for TLR
            elif func == FUNC_TYPE.Recognize_TLR and str(blob_list[-1]) in end_layers[FUNC_TYPE.Recognize_TLR]:
                constructor_for_tlr += "  FC_BLOB_IDX = {};\n".format(len(blob_list) - 1)

        if layer_def.name in end_layer:
            if (func == None and layer_def.name in end_layers[FUNC_TYPE.ComputeDeepConv] + end_layers[FUNC_TYPE.Detect_OD]) or (func == FUNC_TYPE.CFNetProc and layer_def.name in end_layers[func]):
                constructor_data_blobs_by_name += "\n"
            elif layer_def.name in end_layers[FUNC_TYPE.Detect_MD] or layer_def.name in end_layers[FUNC_TYPE.MDNetProc]:
                # for maskrcnn
                mask_class = ["pedestrian", "rider", "bicycle", "motorbike", "car", "van", "truck", "excavator", "forklift", "crane", "vehicle", "bus"]     
                class_names = parse_param_str(proto.class_names)
                if func == FUNC_TYPE.MDNetProc:
                    constructor_has_mask = "\n"
                    for i in xrange(len(class_names)):
                        has_mask = False
                        for class_name in mask_class:
                            if class_name in class_names[i] or class_names[i] in class_name:
                                has_mask = True
                                break
                        if has_mask:
                            constructor_has_mask += "  has_mask[{}] = true;\n".format(class_type_map[class_names[i]])
                        else:
                            constructor_has_mask += "  has_mask[{}] = false;\n".format(class_type_map[class_names[i]])
                else:
                    constructor_has_mask += "  const bool _has_mask[] = { "        
                    for i in xrange(len(class_names)):
                        has_mask = False
                        for class_name in mask_class:
                            if class_name in class_names[i] or class_names[i] in class_name:
                                has_mask = True
                                break
                        if has_mask:
                            constructor_has_mask += "true"
                        else:
                            constructor_has_mask += "false"
                        if i < len(class_names) - 1:
                            constructor_has_mask += ", "
                    constructor_has_mask += " };\n"
                    constructor_has_mask += "  has_mask.assign(_has_mask, _has_mask + {});\n".format(len(class_names))
            elif layer_def.name in end_layers[FUNC_TYPE.Recognize_FPR]:
                # for FPR
                mask_class = ["pedestrian"]
                class_names = parse_param_str(proto.class_names)
                constructor_checks_fp += "  const bool_t _checks_fp[] = { "
                for i in xrange(len(class_names)):
                    checks_fp = False
                    for class_name in mask_class:
                        if class_name in class_names[i] or class_names[i] in class_name:
                            checks_fp = True
                            break
                    if checks_fp:
                        constructor_checks_fp += "true"
                    else:
                        constructor_checks_fp += "false"
                    if i < len(class_names) - 1:
                        constructor_checks_fp += ", "
                constructor_checks_fp += " };\n"
                constructor_checks_fp += "  checks_fp.assign(_checks_fp, _checks_fp + {});\n".format(len(class_names))

            end_idx = idx
            break
    
    if not skip_weights: 
        close_wt(fw)
        if is_int8: close_wt(fw_int8)
    wt_def = make_wt_def(net_name, wt_count) 
    if is_int8: wt_def += make_wt_def(net_name, wt_count_int8, "_int8")

    blob_list_index = {}
    for blob in blob_list:
        blob_list_index[blob] = blob_list.index(blob)  
    layer_list_index = {}
    for layer in layer_list:
        layer_list_index[layer] = layer_list.index(layer)

    constructor += impl_head + " {\n"

    constructor += "  data_blobs.resize({}, {{}});\n".format(len(blob_list))
    constructor += "  layer_blobs.resize({});\n".format(len(layer_list))    
    # For initialize Tensors of data, layer blobs
    constructor += "  for (auto &blob : data_blobs) {\n"
    constructor += "    blob.Initialize();\n"
    constructor += "  }\n"
    constructor += "  for (auto &layer_blob : layer_blobs) {\n"
    constructor += "    for (auto &blob : layer_blob) {\n"
    constructor += "      blob.Initialize();\n"
    constructor += "    }\n"
    constructor += "  }\n"

    if start_layer == start_layers[FUNC_TYPE.Segment_LBF] or start_layer == start_layers[FUNC_TYPE.Segment_LB]:
        constructor += "  typemaps_num_ = 6;\n"
        constructor += "  boundary_typemaps_num_ = 3;\n"
        constructor += "  type_map_vector_.resize(typemaps_num_);\n"
        constructor += "  boundary_type_map_vector_.resize(boundary_typemaps_num_);\n"

    if start_layer == start_layers[FUNC_TYPE.Segment_LD] or start_layer == start_layers[FUNC_TYPE.Segment_LF]:
        constructor += "  typemaps_num_ = 6;\n"
        constructor += "  type_map_vector_.resize(typemaps_num_);\n"


    constructor += "  unsigned int offset = 0;\n" 
    if is_int8:
        constructor += "  unsigned int offset_int8 = 0;\n"
    
    constructor += constructor_layer_init

    if is_int8_qt:
        if "concat_quantized_feature1" in quantize_info["fl_data"]:
                concat_quantized_feature1_min=7;
                for blob in blob_list:
                    if blob=="upsample3" or blob=="eltSum2" or blob=="quantized_feature1" or blob=="fc" or blob=="fc_reshape":
                        if quantize_info["fl_data"][blob]<concat_quantized_feature1_min:
                            concat_quantized_feature1_min=quantize_info["fl_data"][blob]

                for blob in blob_list:
                    if blob=="upsample3" or blob=="eltSum2" or blob=="quantized_feature1" or blob=="fc" or blob=="fc_reshape" or blob=="concat_quantized_feature1":
                       quantize_info["fl_data"][blob]=concat_quantized_feature1_min
        for blob in blob_list:
            if blob in quantize_info["fl_data"] and quantize_info["fl_data"][blob] != "INVALID_SHIFT" and "split" not in blob:
                constructor += "  SET_SHIFTN_D({}, {}); // {}\n".format(blob_list_index[blob], quantize_info["fl_data"][blob], blob)    
        
    constructor += constructor_class_map
    constructor += constructor_has_attribute
    constructor += constructor_has_mask
    constructor += constructor_for_tsr
    constructor += constructor_for_tlr
    constructor += constructor_checks_fp
    if constructor_data_blobs_by_name != "":
        constructor_data_blobs_by_name += "  "
        for blob in blob_list:
            constructor_data_blobs_by_name += "data_blobs_by_name[\"{}\"] = &data_blobs[{}]; ".format(blob,  blob_list_index[blob])
        constructor_data_blobs_by_name += "\n"
        constructor += constructor_data_blobs_by_name
    constructor += "}\n\n"
    constructor += concrete_head.replace("PLATFORM", "CPU") + " {\n"
    constructor += concrete_body.replace("PLATFORM", "CPU")
    constructor += constructor_binary_fc
    constructor += "}\n"
    constructor += "REGISTER_NET({}_CPU);\n\n".format(net_name)
    constructor += "#ifdef USE_INT\n"
    constructor += concrete_head.replace("PLATFORM", "INT") + " {\n"
    if is_int8:
        constructor += concrete_body.replace("PLATFORM", "INT")
        constructor += constructor_binary_fc
        constructor += "}\n"
        constructor += "REGISTER_NET({}_INT);\n".format(net_name)
    else:
        constructor += "  NOT_IMPLEMENT_ERROR;\n"
        constructor += "}\n"
    constructor += "#endif\n\n"
    constructor += "#ifdef USE_GPU\n"
    constructor += concrete_head.replace("PLATFORM", "GPU") + " {\n"
    constructor += concrete_body.replace("PLATFORM", "GPU")
    constructor += constructor_binary_fc
    constructor += "}\n"
    constructor += "REGISTER_NET({}_GPU);\n".format(net_name)
    constructor += "#ifdef USE_FP16\n"
    constructor += concrete_head.replace("PLATFORM", "FP16") + " {\n"
    constructor += concrete_body.replace("PLATFORM", "FP16")
    constructor += constructor_binary_fc
    constructor += "}\n"
    constructor += "REGISTER_NET({}_FP16);\n".format(net_name)
    constructor += "#endif\n"
    constructor += "#endif\n\n"

    return constructor, wt_def, weights_file_list, blob_list, blob_list_index, layer_list, layer_list_index

def make_function(net, proto, net_name, func, blob_list, blob_list_index, layer_list, layer_list_index, god_base):
    def check_split(net, idx, blob_name):
        for blob in net.blobs:
            if blob_name in blob and "split" in blob:
                return True                
        return False           

    def get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto):
        bottom0 = get_bottom(layer_def, blob_list_index, 0)
        bottom0_name = None
        if func == FUNC_TYPE.RPNetProc:
            no_blob = True
            for prev_idx in xrange(start_idx, idx):
                prev_layer_def = get_layer_def(net, proto, prev_idx)
                if layer_def.bottom[0] in prev_layer_def.top[0]:
                    no_blob = False
                    break
            if no_blob:
                bottom0 = "feat_maps->Get(\"{}\")".format(layer_def.bottom[0])
                bottom0_name = layer_def.bottom[0]
        return bottom0, bottom0_name

    roi_pooling_type = ["MAX_POOL", "AVE_POOL", "ALIGN_MAX_POOL"]

    skip_sigmoid_layer = ["af_rpn_cls_prob"]
    disable_layers_for_speedup_rpn = ["rpn_bbox_pred", "rpn_cls_score_reshape", "rpn_cls_prob_reshape", "rpn_cls_prob", "seg_prob", "seg-seg_prob", "fsd-seg_prob", "fsd_seg_prob", "lane-seg_prob", "lane_seg_prob"]
    fold_layers_for_speedup_rpn = ["seg_score_reshape", "seg-seg_score_reshape", "fsd-seg_score_reshape", "fsd_seg_score_reshape", "lane-seg_score_reshape", "lane_seg_score_reshape"]    

    func_name = func_names[func]
    if func == FUNC_TYPE.CFNetProc or func == FUNC_TYPE.RPNetProc or func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
        function = "sv_err_t {}::{} {{\n".format(net_name, functions[func])
    else:
        function = "void {}::{} {{\n".format(net_name, functions[func])
    start_layer = start_layers[func]
    end_layer = end_layers[func]
    
    start_idx = -1
    end_idx = -1
    for idx, layer in enumerate(net.layers):
        layer_def = get_layer_def(net, proto, idx)
        if layer_def is None:
            continue
        if start_idx == -1 and layer_def.name in start_layer:
            start_idx = idx
            function += "  EVT_START;\n"
        if start_idx == -1:
            continue

        if is_reshape(net.layers, idx) or is_softmax(net.layers, idx) or is_sigmoid(net.layers, idx):
            if not check_split(net, idx, layer_def.bottom[0]):
                blob_list_index[layer_def.top[0]] = blob_list_index[layer_def.bottom[0]]

        if layer_def.name in skip_sigmoid_layer and layer_def.type == "Sigmoid":
            continue
        layer_idx = layer_list_index[layer_def.name]

        fc2conv = ""
        if func == FUNC_TYPE.Detect_OD or func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR or func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
            fc2conv = ", FC2CONV"            

        if is_disable_layer(layer_def, net, disable_layers_for_speedup_rpn):
            function += "#ifndef SPEEDUP_RPN\n"

        function += "  // {}\n".format(layer_def.name)    

        unknown_layer = False
        if is_roidata(net.layers, idx):
            if layer_def.name == start_layers[FUNC_TYPE.Segment_LD][1]:
                function += "  const float_t means[4] = { 0.010857, 0.021958, 0.022039, 0.006659 };\n"
            else:
            	function += "  const float_t means[3] = {{ {}, {}, {} }};\n".format(layer_def.roi_data_param.mean0, layer_def.roi_data_param.mean1, layer_def.roi_data_param.mean2)
            if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                function += "  if (!enabled_tiling) {\n"                      
                function += "    i2t->set_input(image, options.DET_RESIZED_HEIGHT, options.DET_RESIZED_WIDTH, options.DET_ROI[0], options.DET_ROI[1], options.DET_ROI[2], options.DET_ROI[3], {}, means, {}, det_crop_x, det_crop_y, det_crop_w, det_crop_h, det_resized_width, det_resized_height, det_scale_x, det_scale_y, options.ROUND_MULTIPLE_RATIO, options.FORCED_ASPECT_RATIO);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))                 
                function += "  } else {\n"
                function += "    i2t->set_input_tiles(image, tile_resized_height, tile_resized_width, tile_height, det_roi_ls, det_roi_ts, det_roi_rs, det_roi_bs, {}, means, {}, det_crop_xs, det_crop_ys, det_crop_ws, det_crop_hs, det_resized_xs, det_resized_widths, det_resized_heights, det_scale_xs, det_scale_ys, options.ROUND_MULTIPLE_RATIO, options.FORCED_ASPECT_RATIO);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
                function += "    det_crop_x = det_crop_xs[0];\n"
                function += "    det_crop_y = det_crop_ys[0];\n"
                function += "    det_crop_w = det_crop_ws[0];\n"
                function += "    det_crop_h = det_crop_hs[0];\n"
                function += "    det_resized_width = det_resized_widths[0];\n"
                function += "    det_resized_height = det_resized_heights[0];\n"
                function += "    det_scale_x = det_scale_xs[0];\n"
                function += "    det_scale_y = det_scale_ys[0];\n"
                function += "  }\n"
                if func == FUNC_TYPE.CFNetProc:
                    function += "  pimage = &image;\n"
            elif func == FUNC_TYPE.Segment_FSD:
                function += "  i2t->set_input(image, options.SEG_FSD_RESIZED_HEIGHT, options.SEG_FSD_RESIZED_WIDTH, options.SEG_FSD_ROI[0], options.SEG_FSD_ROI[1], options.SEG_FSD_ROI[2], options.SEG_FSD_ROI[3], {}, means, {}, seg_fsd_crop_x, seg_fsd_crop_y, seg_fsd_crop_w, seg_fsd_crop_h, seg_fsd_resized_width, seg_fsd_resized_height, seg_fsd_scale_x, seg_fsd_scale_y);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Segment_LSD:
                function += "  i2t->set_input(image, options.SEG_LSD_RESIZED_HEIGHT, options.SEG_LSD_RESIZED_WIDTH, options.SEG_LSD_ROI[0], options.SEG_LSD_ROI[1], options.SEG_LSD_ROI[2], options.SEG_LSD_ROI[3], {}, means, {}, seg_lsd_crop_x, seg_lsd_crop_y, seg_lsd_crop_w, seg_lsd_crop_h, seg_lsd_resized_width, seg_lsd_resized_height, seg_lsd_scale_x, seg_lsd_scale_y);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Segment_LD:
                function += "  Tensor&  line_seg_map = *ld_maps[0];\n"
                function += "  Tensor&  line_cluster_map = *ld_maps[1];\n"
                function += "  Tensor&  vp_seg_map = *ld_maps[2];\n"
                function += "  Tensor&  line_type_val = *ld_maps[3];\n"
                function += "  Tensor&  line_type_shape = *ld_maps[4];\n"
                function += "  Tensor&  line_type_sd = *ld_maps[5];\n"
                function += "  Tensor&  line_type_pos = *ld_maps[6];\n"
                function += "  Tensor&  line_type_color = *ld_maps[7];\n"      
                function += "  Tensor&  line_type_bicycle = *ld_maps[8];\n"
                function += "  float_t& cluster_min_val = *cluster_minmax_vals[0];\n"
                function += "  float_t& cluster_max_val = *cluster_minmax_vals[1];\n"
                function += "  cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["quantized_feature1"].blobs[0].data[0]/net.layer_dict["quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["quantized_feature1"].blobs[0].data[1]/net.layer_dict["quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)    
                function += "  int32_t LANENET_WIDTH = options_.LANENET_WIDTH;\n"
                function += "  int32_t LANENET_HEIGHT = options_.LANENET_HEIGHT;\n"
                function += "  std::vector<float_t> LANENET_ROI_RATIO = options_.IMAGE2LANENET_ROI;\n"
                function += "  image2tensor_->set_input(image, LANENET_HEIGHT, LANENET_WIDTH, LANENET_ROI_RATIO[0], LANENET_ROI_RATIO[1], LANENET_ROI_RATIO[2], LANENET_ROI_RATIO[3], {}, means, {}, image2net_crop_x_, image2net_crop_y_, image2net_crop_w_, image2net_crop_h_, net_width_, net_height_, image2net_scale_x_, image2net_scale_y_, true);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Segment_LF:
                function += "  Tensor&  line_seg_map = *ld_maps[0];\n"
                function += "  Tensor&  line_cluster_map = *ld_maps[1];\n"
                function += "  Tensor&  vp_seg_map = *ld_maps[2];\n"
                function += "  Tensor&  line_type_val = *ld_maps[3];\n"
                function += "  Tensor&  line_type_shape = *ld_maps[4];\n"
                function += "  Tensor&  line_type_sd = *ld_maps[5];\n"
                function += "  Tensor&  line_type_pos = *ld_maps[6];\n"
                function += "  Tensor&  line_type_color = *ld_maps[7];\n"  
                function += "  Tensor&  freespace_seg_map = *ld_maps[8];\n"
                function += "  float_t& cluster_min_val = *cluster_minmax_vals[0];\n"
                function += "  float_t& cluster_max_val = *cluster_minmax_vals[1];\n"
                function += "  cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[0]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[1]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)    
                function += "  int32_t LFNET_WIDTH = options_.LFNET_WIDTH;\n"
                function += "  int32_t LFNET_HEIGHT = options_.LFNET_HEIGHT;\n"
                function += "  std::vector<float_t> LFNET_ROI_RATIO = options_.IMAGE2LFNET_ROI;\n"
                function += "  image2tensor_->set_input(image, LFNET_HEIGHT, LFNET_WIDTH, LFNET_ROI_RATIO[0], LFNET_ROI_RATIO[1], LFNET_ROI_RATIO[2], LFNET_ROI_RATIO[3], {}, means, {}, image2net_crop_x_, image2net_crop_y_, image2net_crop_w_, image2net_crop_h_, net_width_, net_height_, image2net_scale_x_, image2net_scale_y_, true);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Segment_LB:
                function += "  Tensor&  line_seg_map = *ld_maps[0];\n"
                function += "  Tensor&  line_cluster_map = *ld_maps[1];\n"
                function += "  Tensor&  vp_seg_map = *ld_maps[2];\n"
                function += "  Tensor&  line_type_val = *ld_maps[3];\n"
                function += "  Tensor&  line_type_shape = *ld_maps[4];\n"
                function += "  Tensor&  line_type_sd = *ld_maps[5];\n"
                function += "  Tensor&  line_type_pos = *ld_maps[6];\n"
                function += "  Tensor&  line_type_color = *ld_maps[7];\n"
                function += "  Tensor&  boundary_seg_map = *ld_maps[8];\n"
                function += "  Tensor&  boundary_cluster_map = *ld_maps[9];\n"
                function += "  Tensor&  boundary_type_val = *ld_maps[10];\n"
                function += "  Tensor&  boundary_type_shape = *ld_maps[11];\n"
                function += "  Tensor&  boundary_type_pos = *ld_maps[12];\n"
                function += "  float_t& cluster_min_val = *cluster_minmax_vals[0];\n"
                function += "  float_t& cluster_max_val = *cluster_minmax_vals[1];\n"
                function += "  float_t& boundary_cluster_min_val = *cluster_minmax_vals[2];\n"
                function += "  float_t& boundary_cluster_max_val = *cluster_minmax_vals[3];\n"
                function += "  cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[0]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[1]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)    
                function += "  boundary_cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["boundary_quantized_feature1"].blobs[0].data[0]/net.layer_dict["boundary_quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  boundary_cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["boundary_quantized_feature1"].blobs[0].data[1]/net.layer_dict["boundary_quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)  
                function += "  int32_t LBFNET_WIDTH = options_.LBNET_WIDTH;\n"
                function += "  int32_t LBFNET_HEIGHT = options_.LBNET_HEIGHT;\n"
                function += "  std::vector<float_t> LBNET_ROI_RATIO = options_.IMAGE2LBNET_ROI;\n"
                function += "  image2tensor_->set_input(image, LBNET_HEIGHT, LBNET_WIDTH, LBNET_ROI_RATIO[0], LBNET_ROI_RATIO[1], LBFNET_ROI_RATIO[2], LBNET_ROI_RATIO[3], {}, means, {}, image2net_crop_x_, image2net_crop_y_, image2net_crop_w_, image2net_crop_h_, net_width_, net_height_, image2net_scale_x_, image2net_scale_y_, true);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Segment_LBF:
                function += "  Tensor&  line_seg_map = *ld_maps[0];\n"
                function += "  Tensor&  line_cluster_map = *ld_maps[1];\n"
                function += "  Tensor&  vp_seg_map = *ld_maps[2];\n"
                function += "  Tensor&  line_type_val = *ld_maps[3];\n"
                function += "  Tensor&  line_type_shape = *ld_maps[4];\n"
                function += "  Tensor&  line_type_sd = *ld_maps[5];\n"
                function += "  Tensor&  line_type_pos = *ld_maps[6];\n"
                function += "  Tensor&  line_type_color = *ld_maps[7];\n"  
                function += "  Tensor&  freespace_seg_map = *ld_maps[8];\n"
                function += "  Tensor&  boundary_seg_map = *ld_maps[9];\n"
                function += "  Tensor&  boundary_cluster_map = *ld_maps[10];\n"
                function += "  Tensor&  boundary_type_val = *ld_maps[11];\n"
                function += "  Tensor&  boundary_type_shape = *ld_maps[12];\n"
                function += "  Tensor&  boundary_type_pos = *ld_maps[13];\n"
                function += "  float_t& cluster_min_val = *cluster_minmax_vals[0];\n"
                function += "  float_t& cluster_max_val = *cluster_minmax_vals[1];\n"
                function += "  float_t& boundary_cluster_min_val = *cluster_minmax_vals[2];\n"
                function += "  float_t& boundary_cluster_max_val = *cluster_minmax_vals[3];\n"
                function += "  cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[0]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["ld_quantized_feature1"].blobs[0].data[1]/net.layer_dict["ld_quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)    
                function += "  boundary_cluster_min_val = {};\n"\
                     .format(round(net.layer_dict["boundary_quantized_feature1"].blobs[0].data[0]/net.layer_dict["boundary_quantized_feature1"].blobs[1].data[0]*100.0)/100.0-2)    
                function += "  boundary_cluster_max_val = {};\n"\
                     .format(round(net.layer_dict["boundary_quantized_feature1"].blobs[0].data[1]/net.layer_dict["boundary_quantized_feature1"].blobs[1].data[0]*100.0)/100.0+2)  
                function += "  int32_t LBFNET_WIDTH = options_.LBFNET_WIDTH;\n"
                function += "  int32_t LBFNET_HEIGHT = options_.LBFNET_HEIGHT;\n"
                function += "  std::vector<float_t> LBFNET_ROI_RATIO = options_.IMAGE2LBFNET_ROI;\n"
                function += "  image2tensor_->set_input(image, LBFNET_HEIGHT, LBFNET_WIDTH, LBFNET_ROI_RATIO[0], LBFNET_ROI_RATIO[1], LBFNET_ROI_RATIO[2], LBFNET_ROI_RATIO[3], {}, means, {}, image2net_crop_x_, image2net_crop_y_, image2net_crop_w_, image2net_crop_h_, net_width_, net_height_, image2net_scale_x_, image2net_scale_y_, true);\n" \
                    .format(layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Recognize_TSR:
                function += "  i2t->set_input(image, {}, {}, tsr_roi.x1 / image.width, tsr_roi.y1 / image.height, tsr_roi.x2 / image.width, tsr_roi.y2 / image.height, {}, means, {}, det_crop_x, det_crop_y, det_crop_w, det_crop_h, det_resized_width, det_resized_height, det_scale_x, det_scale_y, true, true);\n" \
                    .format(layer_def.roi_data_param.scale_min[0], layer_def.roi_data_param.scale_max[0], layer_def.roi_data_param.multiple, get_top(layer_def, blob_list_index, 0))
            elif func == FUNC_TYPE.Recognize_TLR:
                shape_ = proto.input_shape._values[0].dim 
                function += "  float_t roi_w = {};\n".format(shape_[3])
                function += "  float_t roi_h = tlr_roi.height() * (roi_w / tlr_roi.width());\n"
                function += "  i2t->set_input(image, roi_w, roi_h, tlr_roi.x1 / image.width, tlr_roi.y1 / image.height, tlr_roi.x2 / image.width, tlr_roi.y2 / image.height, 1, means, {}, det_crop_x, det_crop_y, det_crop_w, det_crop_h, det_resized_width, det_resized_height, det_scale_x, det_scale_y, true, true, {}, {});\n" \
                    .format(get_top(layer_def, blob_list_index, 0), shape_[3], shape_[2])
            else:
                assert(0)
            function += "  image_width = image.width;\n"
            function += "  image_height = image.height;\n"

        elif is_conv(net.layers, idx):
            stride = 1
            if len(layer_def.convolution_param.stride) > 0: 
                stride = layer_def.convolution_param.stride[0]
            stride_h = -1
            stride_w = -1
            if layer_def.convolution_param.stride_h!=0 or layer_def.convolution_param.stride_w!=0:
                stride_h = layer_def.convolution_param.stride_h
                stride_w = layer_def.convolution_param.stride_w
            else:
                stride_h=stride
            pad = 0
            if len(layer_def.convolution_param.pad) > 0:
                pad = layer_def.convolution_param.pad[0]           
            pad_h=-1
            pad_w=-1
            if layer_def.convolution_param.pad_h != 0 or layer_def.convolution_param.pad_w != 0:
                pad_h = layer_def.convolution_param.pad_h
                pad_w = layer_def.convolution_param.pad_w
            else:
                pad_h=pad    
            kernel_size = 1
            if len(layer_def.convolution_param.kernel_size) > 0:
                kernel_size = layer_def.convolution_param.kernel_size[0]
            dilation = 1
            if len(layer_def.convolution_param.dilation) > 0:
                dilation = layer_def.convolution_param.dilation[0]

            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)

            if kernel_size > 1:
                if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                    function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                elif func == FUNC_TYPE.RPNetProc:
                    if bottom0_name == None:
                        function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                    else:
                        function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
            if pad_w != -1:
                function += '  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {}, {}, {});\n' \
                    .format(bottom0, layer_idx, layer_idx, stride_h, pad_h, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0), stride_w, pad_w)
            elif stride_w!=-1:
                function += '  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {}, {});\n' \
                    .format(bottom0, layer_idx, layer_idx, stride_h, pad, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0), stride_w)
            else:
                function += "  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {});\n" \
                  .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0))

        elif is_conv_relu(net.layers, idx):
            stride = 1
            if len(layer_def.convolution_param.stride) > 0: 
                stride = layer_def.convolution_param.stride[0]
            stride_h = -1
            stride_w = -1
            if layer_def.convolution_param.stride_h!=0 or layer_def.convolution_param.stride_w!=0:
                stride_h = layer_def.convolution_param.stride_h
                stride_w = layer_def.convolution_param.stride_w
            else:
                stride_h=stride
            pad = 0
            if len(layer_def.convolution_param.pad) > 0:
                pad = layer_def.convolution_param.pad[0]           
            pad_h=-1
            pad_w=-1
            if layer_def.convolution_param.pad_h != 0 or layer_def.convolution_param.pad_w != 0:
                pad_h = layer_def.convolution_param.pad_h
                pad_w = layer_def.convolution_param.pad_w
            else:
                pad_h=pad               
            kernel_size = 1
            if len(layer_def.convolution_param.kernel_size) > 0:
                kernel_size = layer_def.convolution_param.kernel_size[0]
            dilation = 1
            if len(layer_def.convolution_param.dilation) > 0:
                dilation = layer_def.convolution_param.dilation[0]

            if net.layers[idx + 1].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 1)
            elif net.layers[idx + 2].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 2)
            elif net.layers[idx + 3].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 3)
            else:
                raise ValueError("net.layers[idx + 1], [idx + 2], or [idx + 3] should be a ReLU layer")

            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)

            if layer_def.convolution_param.bias_term == True and get_top(layer_def, blob_list_index, 0) == get_bottom(relu_layer_def, blob_list_index, 0):
                if kernel_size > 1:
                    if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                        function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                    elif func == FUNC_TYPE.RPNetProc:
                        if bottom0_name == None:
                            function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                        else:
                            function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                if pad_w!=-1:
                    function += "  cnn->conv_bias_relu({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {}, {}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride_h, pad_h, layer_def.convolution_param.group, dilation, get_top(layer_def, blob_list_index, 0), stride_w, pad_w)
                elif stride_w!=-1:
                    function += "  cnn->conv_bias_relu({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride_h, pad, layer_def.convolution_param.group, dilation, get_top(layer_def, blob_list_index, 0), stride_w)
                else:
                    function += "  cnn->conv_bias_relu({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, get_top(layer_def, blob_list_index, 0))
            else:
                if kernel_size > 1:
                    if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                        function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                    elif func == FUNC_TYPE.RPNetProc:
                        if bottom0_name == None:
                            function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                        else:
                            function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)

                if pad_w != -1:
                    function += '  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {}, {}, {});\n' \
                        .format(bottom0, layer_idx, layer_idx, stride_h, pad_h, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0), stride_w, pad_w)
                elif stride_w!=-1:
                    function += '  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {}, {});\n' \
                        .format(bottom0, layer_idx, layer_idx, stride_h, pad, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0), stride_w)
                else:
                    function += "  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0))
                function += "  cnn->relu({}, 0);\n" \
                    .format(get_top(relu_layer_def, blob_list_index, 0))

        elif is_conv_eltwise_relu(net.layers, idx):
            stride = 1
            if len(layer_def.convolution_param.stride) > 0: 
                stride = layer_def.convolution_param.stride[0]
            pad = 0
            if len(layer_def.convolution_param.pad) > 0:
                pad = layer_def.convolution_param.pad[0]               
            kernel_size = 1
            if len(layer_def.convolution_param.kernel_size) > 0:
                kernel_size = layer_def.convolution_param.kernel_size[0]
            dilation = 1
            if len(layer_def.convolution_param.dilation) > 0:
                dilation = layer_def.convolution_param.dilation[0]

            elt_operation = 1
            if net.layers[idx + 1].type == "Eltwise":
                eltwise_layer_def = get_layer_def(net, proto, idx + 1)
                elt_operation = eltwise_layer_def.eltwise_param.operation
            elif net.layers[idx + 2].type == "Eltwise":
                eltwise_layer_def = get_layer_def(net, proto, idx + 2)
                elt_operation = eltwise_layer_def.eltwise_param.operation
            elif net.layers[idx + 3].type == "Eltwise":
                eltwise_layer_def = get_layer_def(net, proto, idx + 3)
                elt_operation = eltwise_layer_def.eltwise_param.operation
            elif net.layers[idx + 1].type == "EReLU":
                eltwise_layer_def = get_layer_def(net, proto, idx + 1)
            elif net.layers[idx + 2].type == "EReLU":
                eltwise_layer_def = get_layer_def(net, proto, idx + 2)
            elif net.layers[idx + 3].type == "EReLU":
                eltwise_layer_def = get_layer_def(net, proto, idx + 3)
            else:
                raise ValueError("net.layers[idx + 1], [idx + 2], or [idx + 3] should be an Eltwise layer")
            if net.layers[idx + 1].type == "Eltwise" and net.layers[idx + 2].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 2)
            elif net.layers[idx + 2].type == "Eltwise" and net.layers[idx + 3].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 3)
            elif net.layers[idx + 3].type == "Eltwise" and net.layers[idx + 4].type == "ReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 4)
            elif net.layers[idx + 1].type == "EReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 1)
            elif net.layers[idx + 2].type == "EReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 2)
            elif net.layers[idx + 3].type == "EReLU":
                relu_layer_def = get_layer_def(net, proto, idx + 3)
            else:
                raise ValueError("net.layers[idx + 2], [idx + 3], or [idx + 4] should be a ReLU layer")

            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)

            if layer_def.convolution_param.bias_term == True and elt_operation == 1 and len(eltwise_layer_def.bottom) == 2 and (get_top(layer_def, blob_list_index, 0) == get_bottom(eltwise_layer_def, blob_list_index, 0) or get_top(layer_def, blob_list_index, 0) == get_bottom(eltwise_layer_def, blob_list_index, 1)) and (get_top(eltwise_layer_def, blob_list_index, 0) == get_bottom(relu_layer_def, blob_list_index, 0) or get_top(eltwise_layer_def, blob_list_index, 0) == get_top(relu_layer_def, blob_list_index, 0)):
                if get_top(layer_def, blob_list_index, 0) == get_bottom(eltwise_layer_def, blob_list_index, 0):
                    if kernel_size > 1:
                        if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                            function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                        elif func == FUNC_TYPE.RPNetProc:
                            if bottom0_name == None:
                                function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                            else:
                                function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                    function += "  cnn->conv_bias_add_relu({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, get_bottom(eltwise_layer_def, blob_list_index, 1), get_top(relu_layer_def, blob_list_index, 0))
                else:
                    if kernel_size > 1:
                        if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                            function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                        elif func == FUNC_TYPE.RPNetProc:
                            if bottom0_name == None:
                                function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                            else:
                                function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                    function += "  cnn->conv_bias_add_relu({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {}, {});\n" \
                        .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, get_bottom(eltwise_layer_def, blob_list_index, 0), get_top(relu_layer_def, blob_list_index, 0))
            else:
                if kernel_size > 1:
                    if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                        function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                    elif func == FUNC_TYPE.RPNetProc:
                        if bottom0_name == None:
                            function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                        else:
                            function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                function += "  cnn->conv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {}, {:d}, {});\n" \
                    .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, dilation, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0))
                if len(eltwise_layer_def.bottom) == 3:
                    function += "  cnn->eltwise_sum_3way({}, {}, {}, {});\n" \
                        .format(get_bottom(eltwise_layer_def, blob_list_index, 0), get_bottom(eltwise_layer_def, blob_list_index, 1), get_bottom(eltwise_layer_def, blob_list_index, 2), get_top(eltwise_layer_def, blob_list_index, 0))
                else:
                    function += "  cnn->eltwise_sum({}, {}, {});\n" \
                        .format(get_bottom(eltwise_layer_def, blob_list_index, 0), get_bottom(eltwise_layer_def, blob_list_index, 1), get_top(eltwise_layer_def, blob_list_index, 0))
                    for eltwise_bottom in xrange(2, len(eltwise_layer_def.bottom)):
                        function += "  cnn->eltwise_sum({}, {}, {});\n" \
                            .format(get_top(eltwise_layer_def, blob_list_index, 0), get_bottom(eltwise_layer_def, blob_list_index, eltwise_bottom), get_top(eltwise_layer_def, blob_list_index, 0))
                function += "  cnn->relu({}, 0);\n" \
                    .format(get_top(relu_layer_def, blob_list_index, 0))

        elif is_deconv(net.layers, idx):
            stride = 1
            if len(layer_def.convolution_param.stride) > 0:
                stride = layer_def.convolution_param.stride[0]
            pad = 0
            if len(layer_def.convolution_param.pad) > 0:
                pad = layer_def.convolution_param.pad[0]
            kernel_size = 1
            if len(layer_def.convolution_param.kernel_size) > 0:
                kernel_size = layer_def.convolution_param.kernel_size[0]

            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)

            if stride == 2 and kernel_size == 2 and layer_def.convolution_param.weight_filler.type == "constant" and layer_def.convolution_param.bias_term == False and net.blobs[layer_def.bottom[0]].channels == net.blobs[layer_def.top[0]].channels and layer_def.convolution_param.group == net.blobs[layer_def.top[0]].channels:
                function += "  cnn->resize({}, 2, 2, 0, {});\n" \
                    .format(bottom0, get_top(layer_def, blob_list_index, 0))
            elif stride == 2 and kernel_size == 4 and layer_def.convolution_param.weight_filler.type == "bilinear" and layer_def.convolution_param.bias_term == False and net.blobs[layer_def.bottom[0]].channels == net.blobs[layer_def.top[0]].channels and layer_def.convolution_param.group == net.blobs[layer_def.top[0]].channels:
                if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                    function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                elif func == FUNC_TYPE.RPNetProc:
                    if bottom0_name == None:
                        function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                    else:
                        function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                function += "  cnn->resize({}, 2, 2, 1, {});\n" \
                    .format(bottom0, get_top(layer_def, blob_list_index, 0))
            elif stride == 2 and kernel_size == 2 and layer_def.convolution_param.weight_filler.type == "constant" and layer_def.convolution_param.bias_term == False and net.blobs[layer_def.bottom[0]].channels == net.blobs[layer_def.top[0]].channels * 2 and layer_def.convolution_param.group == net.blobs[layer_def.top[0]].channels:
                function += "  cnn->resize_padd({}, 2, 2, 0, {});\n" \
                    .format(bottom0, get_top(layer_def, blob_list_index, 0))
            elif stride == 2 and kernel_size == 4 and layer_def.convolution_param.weight_filler.type == "bilinear" and layer_def.convolution_param.bias_term == False and net.blobs[layer_def.bottom[0]].channels == net.blobs[layer_def.top[0]].channels * 2 and layer_def.convolution_param.group == net.blobs[layer_def.top[0]].channels:
                if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                    function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                elif func == FUNC_TYPE.RPNetProc:
                    if bottom0_name == None:
                        function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                    else:
                        function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                function += "  cnn->resize_padd({}, 2, 2, 1, {});\n" \
                    .format(bottom0, get_top(layer_def, blob_list_index, 0))
            else:
                if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                    function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
                elif func == FUNC_TYPE.RPNetProc:
                    if bottom0_name == None:
                        function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                    else:
                        function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
                function += "  cnn->deconv({}, layer_blobs[{}][0], layer_blobs[{}][1], {}, {}, {}, {:d}, {});\n" \
                    .format(bottom0, layer_idx, layer_idx, stride, pad, layer_def.convolution_param.group, layer_def.convolution_param.bias_term, get_top(layer_def, blob_list_index, 0))

        elif is_upsample(net.layers, idx):
            if layer_def.upsample_param.scale == 2:
                function += "  cnn->resize({}, 2, 2, 0, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_top(layer_def, blob_list_index, 0))

        elif is_scale(net.layers, idx):
            function += "  cnn->scale({}, layer_blobs[{}][0], layer_blobs[{}][1], {:d}, {});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), layer_idx, layer_idx, layer_def.scale_param.bias_term, get_top(layer_def, blob_list_index, 0))
            
        elif is_crelu(net.layers, idx):
            if net.layers[idx].type == "CReLU":
                crelu_out_layer_def = layer_def
            else:
                crelu_out_layer_def = get_layer_def(net, proto, idx + 2)

            function += "  cnn->crelu({}, {});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), get_top(crelu_out_layer_def, blob_list_index, 0))            

        elif is_relu(net.layers, idx):
            function += "  cnn->relu({}, 0);\n" \
                .format(get_top(layer_def, blob_list_index, 0))            

        elif is_elu(net.layers, idx):
            function += "  cnn->elu({}, 1);\n" \
                .format(get_top(layer_def, blob_list_index, 0))

        elif is_mish(net.layers, idx):
            function += "  cnn->mish({});\n" \
                .format(get_top(layer_def, blob_list_index, 0))

        elif is_erelu(net.layers, idx):
            function += "  cnn->eltwise_sum({}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, 1), get_top(layer_def, blob_list_index, 0))
            function += "  cnn->relu({}, 0);\n" \
                .format(get_top(layer_def, blob_list_index, 0)) 

        elif is_eltwise(net.layers, idx) and layer_def.eltwise_param.operation == 0:
            function += "  cnn->eltwise_mul({}, {}, {});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, 1), get_top(layer_def, blob_list_index, 0))
            for eltwise_bottom in xrange(2, len(layer_def.bottom)):
                function += "  cnn->eltwise_mul({}, {}, {});\n" \
                    .format(get_top(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, eltwise_bottom), get_top(layer_def, blob_list_index, 0))

        elif is_eltwise(net.layers, idx) and layer_def.eltwise_param.operation == 1:
            if len(layer_def.bottom) == 3:
                function += "  cnn->eltwise_sum_3way({}, {}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, 1), get_bottom(layer_def, blob_list_index, 2), get_top(layer_def, blob_list_index, 0))
            else:
                function += "  cnn->eltwise_sum({}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, 1), get_top(layer_def, blob_list_index, 0))
                for eltwise_bottom in xrange(2, len(layer_def.bottom)):
                    function += "  cnn->eltwise_sum({}, {}, {});\n" \
                        .format(get_top(layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, eltwise_bottom), get_top(layer_def, blob_list_index, 0))

        elif is_pooling(net.layers, idx) and layer_def.pooling_param.pool == 0:
            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)
            if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
            elif func == FUNC_TYPE.RPNetProc:
                if bottom0_name == None:
                    function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                else:
                    function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
            function += "  cnn->max_pooling({}, {}, {}, {}, {}, {});\n" \
                .format(bottom0, layer_def.pooling_param.kernel_size, layer_def.pooling_param.kernel_size, layer_def.pooling_param.stride, layer_def.pooling_param.pad, get_top(layer_def, blob_list_index, 0))

        elif is_pooling(net.layers, idx) and layer_def.pooling_param.pool == 1:
            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)
            if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
            elif func == FUNC_TYPE.RPNetProc:
                if bottom0_name == None:
                    function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                else:
                    function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
            if layer_def.pooling_param.global_pooling == True:
                function += "  cnn->ave_pooling({}, {}.d3, {}.d4, {}, {}, {});\n" \
                    .format(bottom0, bottom0, bottom0, layer_def.pooling_param.stride, layer_def.pooling_param.pad, get_top(layer_def, blob_list_index, 0))
            else:
                function += "  cnn->ave_pooling({}, {}, {}, {}, {}, {});\n" \
                    .format(bottom0, layer_def.pooling_param.kernel_size, layer_def.pooling_param.kernel_size, layer_def.pooling_param.stride, layer_def.pooling_param.pad, get_top(layer_def, blob_list_index, 0))

        elif is_stixelpooling(net.layers, idx) and layer_def.stixelpooling_param.pool == 0:
            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)
            if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
            elif func == FUNC_TYPE.RPNetProc:
                if bottom0_name == None:
                    function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                else:
                    function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)
            if god_base:
                function += "  cnn->max_stixelpooling({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, enabled_tiling, det_resized_widths, tile_height);\n" \
                .format(bottom0, np.int(layer_def.stixelpooling_param.global_pooling_h),  layer_def.stixelpooling_param.pad_h,layer_def.stixelpooling_param.stride_h, layer_def.stixelpooling_param.bin_h, layer_def.stixelpooling_param.kernel_h,np.int(layer_def.stixelpooling_param.global_pooling_w),layer_def.stixelpooling_param.pad_w,layer_def.stixelpooling_param.stride_w, layer_def.stixelpooling_param.bin_w,layer_def.stixelpooling_param.kernel_w, get_top(layer_def, blob_list_index, 0))
            else:
                function += "  cnn->max_stixelpooling({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 0);\n" \
                .format(bottom0, np.int(layer_def.stixelpooling_param.global_pooling_h),  layer_def.stixelpooling_param.pad_h,layer_def.stixelpooling_param.stride_h, layer_def.stixelpooling_param.bin_h, layer_def.stixelpooling_param.kernel_h,np.int(layer_def.stixelpooling_param.global_pooling_w),layer_def.stixelpooling_param.pad_w,layer_def.stixelpooling_param.stride_w, layer_def.stixelpooling_param.bin_w,layer_def.stixelpooling_param.kernel_w, get_top(layer_def, blob_list_index, 0))

        elif is_stixelpooling(net.layers, idx) and layer_def.stixelpooling_param.pool == 1:
            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)
            if func == FUNC_TYPE.ComputeDeepConv or func == FUNC_TYPE.CFNetProc:
                function += "  if (enabled_tiling)\n    cnn->set_tile_boundaries({}, det_resized_widths, tile_height);\n".format(bottom0)
            elif func == FUNC_TYPE.RPNetProc:
                if bottom0_name == None:
                    function += "  if (feat_maps->GetTiling())\n    cnn->set_tile_boundaries({}, feat_maps->GetDetResizedWidths(), feat_maps->GetTileHeight());\n".format(bottom0)
                else:
                    function += "  feat_maps->SetTileBoundaries(\"{}\");\n".format(bottom0_name)           
            if god_base:
                function += "  cnn->ave_stixelpooling({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, enabled_tiling, det_resized_widths, tile_height);\n" \
                .format(bottom0, np.int(layer_def.stixelpooling_param.global_pooling_h),  layer_def.stixelpooling_param.pad_h,layer_def.stixelpooling_param.stride_h, layer_def.stixelpooling_param.bin_h, layer_def.stixelpooling_param.kernel_h,np.int(layer_def.stixelpooling_param.global_pooling_w),layer_def.stixelpooling_param.pad_w,layer_def.stixelpooling_param.stride_w, layer_def.stixelpooling_param.bin_w,layer_def.stixelpooling_param.kernel_w, get_top(layer_def, blob_list_index, 0))
            else:
                function += "  cnn->ave_stixelpooling({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 0);\n" \
                .format(bottom0, np.int(layer_def.stixelpooling_param.global_pooling_h),  layer_def.stixelpooling_param.pad_h,layer_def.stixelpooling_param.stride_h, layer_def.stixelpooling_param.bin_h, layer_def.stixelpooling_param.kernel_h,np.int(layer_def.stixelpooling_param.global_pooling_w),layer_def.stixelpooling_param.pad_w,layer_def.stixelpooling_param.stride_w, layer_def.stixelpooling_param.bin_w,layer_def.stixelpooling_param.kernel_w, get_top(layer_def, blob_list_index, 0))

        elif is_concat(net.layers, idx):
            cocnat_inputs = "std::vector<const Tensor*> {"
            for bottom_idx, bottom in enumerate(layer_def.bottom):
                cocnat_inputs += "&{}, ".format(get_bottom(layer_def, blob_list_index, bottom_idx))
            cocnat_inputs = cocnat_inputs[:-2] + "}"                
            if layer_def.name == "ego_out":
                function += "  cnn->concat({}, {}, {});\n".format(cocnat_inputs, layer_def.concat_param.axis, "Ego")
            else:
                function += "  cnn->concat({}, {}, {});\n".format(cocnat_inputs, layer_def.concat_param.axis, get_top(layer_def, blob_list_index, 0))

            if is_grid_conv_end(net.layers, idx):
                function += "#if defined(SPEEDUP_GRIDCONV) && defined(USE_CUDNN)\n"
                function += "  }\n"
                function += "#endif\n"
        elif is_slice(net.layers, idx):
            if is_grid_conv_start(net.layers, idx):
                grid_conv_output_def = get_layer_def(net, proto, idx + 25)
                function += "#if defined(SPEEDUP_GRIDCONV) && defined(USE_CUDNN)\n"
                function += "  if (UseGPU()) {\n"
                function += "  std::vector<const Tensor*> weights, biases;\n"
                for grid_index in xrange(4):
                        function += "  weights.push_back(&layer_blobs[{}][0]); biases.push_back(&layer_blobs[{}][1]);\n".format(layer_idx + 5 + grid_index * 4, layer_idx + 5 + grid_index * 4)
                for grid_index in xrange(4):
                        function += "  weights.push_back(&layer_blobs[{}][0]); biases.push_back(&layer_blobs[{}][1]);\n".format(layer_idx + 6 + grid_index * 4, layer_idx + 6 + grid_index * 4)
                for grid_index in xrange(4):
                        function += "  weights.push_back(&layer_blobs[{}][0]); biases.push_back(&layer_blobs[{}][1]);\n".format(layer_idx + 7 + grid_index * 4, layer_idx + 7 + grid_index * 4)
                for grid_index in xrange(4):
                        function += "  weights.push_back(&layer_blobs[{}][0]); biases.push_back(&layer_blobs[{}][1]);\n".format(layer_idx + 8 + grid_index * 4, layer_idx + 8 + grid_index * 4)
                function += "  cnn->grid_conv({}, weights, biases, 4, 4, 1, 0, 1, 1, 1, {});\n".format(get_bottom(layer_def, blob_list_index, 0), get_top(grid_conv_output_def, blob_list_index, 0))
                function += "  } else {\n"
                function += "#endif\n"

            slice_outputs = "std::vector<Tensor*> {"
            for top_idx, top in enumerate(layer_def.top):       
                slice_outputs += "&{}, ".format(get_top(layer_def, blob_list_index, top_idx))
            slice_outputs = slice_outputs[:-2] + "}"                
            slice_points = "std::vector<int> {"
            if len(layer_def.slice_param.slice_point) == 0:
                slice_points += "-1}"
            else:
                for slice_idx in xrange(len(layer_def.slice_param.slice_point)):                        
                    slice_points += "{}, ".format(layer_def.slice_param.slice_point[slice_idx])
                slice_points = slice_points[:-2] + "}"
            function += "  cnn->slice({}, {}, {}, {});\n".format(get_bottom(layer_def, blob_list_index, 0), layer_def.slice_param.axis, slice_points, slice_outputs)
        elif is_sliceconcat(net.layers, idx):
            slice_outputs = "std::vector<Tensor*> {"
            cocnat_inputs = "std::vector<const Tensor*> {"
            for slice_idx in range(layer_def.slice_concat_param.nslices):
                slice_outputs += "&tmp[{}], ".format(slice_idx)
                cocnat_inputs += "&tmp[{}], ".format(slice_idx)
            slice_outputs = slice_outputs[:-2] + "}"
            cocnat_inputs = cocnat_inputs[:-2] + "}"            
            slice_points = "std::vector<int> {-1}"
            function += "  {{ std::vector<Tensor> tmp({}, {{}}); ".format(int(layer_def.slice_concat_param.nslices))
            function += "cnn->slice({}, {}, {}, {});".format(get_bottom(layer_def, blob_list_index, 0), layer_def.slice_concat_param.slice_axis, slice_points, slice_outputs)
            function += "cnn->concat({}, {}, {}); }}\n".format(cocnat_inputs, layer_def.slice_concat_param.concat_axis, get_top(layer_def, blob_list_index, 0))
        elif is_proposalnway(net.layers, idx):
            scores = ""
            bbox_deltas = ""
            anchors = ""
            feat_stride = ""
            base_size = ""
            min_size = ""
            max_size = ""
            feats = ""
            bbox_pred_weight = ""
            bbox_pred_bias = ""
            for i in xrange(len(layer_def.proposal_nway_param.rpn_option)):
                if i * 2 + 1 < len(layer_def.bottom):
                    proposalnway_bottom_layer_def = get_layer_def_by_name(proto, layer_def.bottom[i * 2 + 1])
                    scores += "&{}, ".format(get_bottom(layer_def, blob_list_index, i * 2))
                    bbox_deltas += "&{}, ".format(get_bottom(layer_def, blob_list_index, i * 2 + 1))
                    if is_disable_layer(proposalnway_bottom_layer_def, net, disable_layers_for_speedup_rpn):
                        bottom0, bottom0_name = get_start_bottom(proposalnway_bottom_layer_def, blob_list_index, func, start_idx, idx, net, proto)
                        feats += "&{}, ".format(bottom0)
                        bbox_pred_weight += "&layer_blobs[{}][0], ".format(layer_list_index[layer_def.bottom[i * 2 + 1]])
                        bbox_pred_bias += "&layer_blobs[{}][1], ".format(layer_list_index[layer_def.bottom[i * 2 + 1]])
                anchors += "&anchors[{}], ".format(i)
                feat_stride += "{}, ".format(layer_def.proposal_nway_param.rpn_option[i].feat_stride)
                base_size += "{}, ".format(layer_def.proposal_nway_param.rpn_option[i].base_size)
                min_size += "{}, ".format(layer_def.proposal_nway_param.rpn_option[i].min_size)
                max_size += "{}, ".format(layer_def.proposal_nway_param.rpn_option[i].max_size)                
            scores = "std::vector<const Tensor*> {" + scores[:-2] + "}"
            bbox_deltas = "std::vector<const Tensor*> {" + bbox_deltas[:-2] + "}"
            anchors = "{" + anchors[:-2] + "}"
            feat_stride = "{" + feat_stride[:-2] + "}"
            base_size = "{" + base_size[:-2] + "}"
            min_size = "{" + min_size[:-2] + "}"
            max_size = "{" + max_size[:-2] + "}"
            if feats != "":
                feats = "std::vector<const Tensor*> {" + feats[:-2] + "}"
                bbox_pred_weight = "std::vector<const Tensor*> {" + bbox_pred_weight[:-2] + "}"
                bbox_pred_bias = "std::vector<const Tensor*> {" + bbox_pred_bias[:-2] + "}"
            
            function += "  const std::vector<const Tensor*> proposal_nway_anchors = {};\n".format(anchors)
            function += "  const std::vector<int32_t> proposal_nway_feat_stride = {};\n".format(feat_stride)
            function += "  const std::vector<int32_t> proposal_nway_base_size = {};\n".format(base_size)
            function += "  const std::vector<int32_t> proposal_nway_min_size = {};\n".format(min_size)
            function += "  const std::vector<int32_t> proposal_nway_max_size = {};\n".format(max_size)
            if func != FUNC_TYPE.RPNetProc:
                function += "  const std::vector<std::vector<float>*> proposal_nway_tile_config = {&det_roi_ls, &det_roi_ts, &det_roi_rs, &det_roi_bs};\n"
            if feats != "":
                function += "#ifndef SPEEDUP_RPN\n"
                function += "  const std::vector<std::vector<const Tensor*> > proposal_nway_sources = {{{}, {}}};\n".format(scores, bbox_deltas)                               
                function += "#else\n"
                function += "  const std::vector<std::vector<const Tensor*> > proposal_nway_sources = {{{}, {}, {}, {}}};\n".format(scores, feats, bbox_pred_weight, bbox_pred_bias)                 
                function += "#endif\n"
            else:
                function += "  const std::vector<std::vector<const Tensor*> > proposal_nway_sources = {{{}, {}}};\n".format(scores, bbox_deltas)
            if func == FUNC_TYPE.RPNetProc:
                function += "  if (!feat_maps->GetTiling()) {\n"            
                function += "    cnn->proposal_nway(proposal_nway_sources, proposal_nway_anchors, proposal_nway_feat_stride, proposal_nway_base_size, proposal_nway_min_size, proposal_nway_max_size, feat_maps->GetDetResizedWidth(), feat_maps->GetDetResizedHeight(), options.PROPOSAL_THRES, proposal_rois);\n"            
                function += "  } else {\n"
                function += "    cnn->proposal_nway_tiles(proposal_nway_sources, proposal_nway_anchors, proposal_nway_feat_stride, proposal_nway_base_size, proposal_nway_min_size, proposal_nway_max_size, feat_maps->GetDetResizedXs(), feat_maps->GetDetResizedWidths(), feat_maps->GetDetResizedHeights(), options.PROPOSAL_THRES, proposal_rois);\n"
                function += "  }\n" 
            else:
                function += "  if (!enabled_tiling) {\n"   
                function += "    cnn->proposal_nway(proposal_nway_sources, proposal_nway_anchors, proposal_nway_feat_stride, proposal_nway_base_size, proposal_nway_min_size, proposal_nway_max_size, det_resized_width, det_resized_height, det_scale_x, det_scale_y, options.PRE_NMS_PROPOSALS, options.POST_NMS_PROPOSALS, options.PROPOSAL_NMS_THRES, rois, {}, options.PROPOSAL_THRES);\n" \
                    .format(get_top(layer_def, blob_list_index, 1))
                function += "  } else {\n"
                function += "    cnn->proposal_nway_tiles(proposal_nway_sources, proposal_nway_anchors, proposal_nway_feat_stride, proposal_nway_base_size, proposal_nway_min_size, proposal_nway_max_size, det_crop_xs, det_crop_ys, det_resized_xs, det_resized_widths, det_resized_heights, det_scale_xs, det_scale_ys, options.PRE_NMS_PROPOSALS, options.POST_NMS_PROPOSALS, options.PROPOSAL_NMS_THRES, rois, {}, proposal_nway_tile_config, tile_filters, options.DET_TILES, options.DET_TILE_UPDATE, options.PROPOSAL_THRES);\n" \
                    .format(get_top(layer_def, blob_list_index, 1))
                function += "  }\n"

        elif is_reshape(net.layers, idx):
            if layer_def.name in fold_layers_for_speedup_rpn or layer_def.name[:-1] in fold_layers_for_speedup_rpn:
                function += "#ifndef SPEEDUP_RPN\n"
                function += "  cnn->reshape({}, {}, {}, {}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], layer_def.reshape_param.shape.dim[1], layer_def.reshape_param.shape.dim[2], layer_def.reshape_param.shape.dim[3], get_top(layer_def, blob_list_index, 0))
                function += "#else\n"
                function += "  cnn->reshape({}, {}, {}, {}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], 1, layer_def.reshape_param.shape.dim[2], layer_def.reshape_param.shape.dim[3], get_top(layer_def, blob_list_index, 0))
                function += "#endif\n"
            elif len(layer_def.reshape_param.shape.dim) == 1:
                function += "  cnn->reshape({}, {}, 1, 1, 1, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], get_top(layer_def, blob_list_index, 0))
            elif len(layer_def.reshape_param.shape.dim) == 2:
                function += "  cnn->reshape({}, {}, {}, 1, 1, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], layer_def.reshape_param.shape.dim[1], get_top(layer_def, blob_list_index, 0))
            elif len(layer_def.reshape_param.shape.dim) == 3:
                function += "  cnn->reshape({}, {}, {}, {}, 1, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], layer_def.reshape_param.shape.dim[1], layer_def.reshape_param.shape.dim[2], get_top(layer_def, blob_list_index, 0))
            else:
                function += "  cnn->reshape({}, {}, {}, {}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.reshape_param.shape.dim[0], layer_def.reshape_param.shape.dim[1], layer_def.reshape_param.shape.dim[2], layer_def.reshape_param.shape.dim[3], get_top(layer_def, blob_list_index, 0))
                

        elif is_roipooling(net.layers, idx):
            bottom0, bottom0_name = get_start_bottom(layer_def, blob_list_index, func, start_idx, idx, net, proto)
            if func == FUNC_TYPE.Detect_ES:
                function += "  cnn->roi_pooling({}, ego_rois, {}, {}, {}, {}{}, {});\n" \
                        .format(get_bottom(layer_def, blob_list_index,0), layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])
            elif func == FUNC_TYPE.Detect_CL:                
                function += "  cnn->roi_pooling({}, closeness_rois, {}, {}, {}, {}{}, {});\n" \
                        .format(get_bottom(layer_def, blob_list_index,0), layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])
            elif func == FUNC_TYPE.Detect_FS:                
                function += "  cnn->roi_pooling({}, fs_rois, {}, {}, {}, {}{}, {});\n" \
                        .format(get_bottom(layer_def, blob_list_index,0), layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])
            elif func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                function += "  cnn->roi_pooling(*feat_maps.at(\"{}\"), rois, {}, {}, {}, {}{}, {});\n" \
                        .format(layer_def.bottom[0], layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])            
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                function += "  cnn->roi_pooling(feat_maps->Get(\"{}\"), rois, {}, {}, {}, {}{}, {});\n" \
                        .format(layer_def.bottom[0], layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])
            else:                
                function += "  cnn->roi_pooling({}, rois, {}, {}, {}, {}{}, {});\n" \
                        .format(bottom0, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, get_top(layer_def, blob_list_index, 0), fc2conv, roi_pooling_type[layer_def.roi_pooling_param.pool])

        elif is_roipooling3way(net.layers, idx):                
            roipooling3way_in0_layer_def = get_layer_def(net, proto, idx + 0)
            roipooling3way_in1_layer_def = get_layer_def(net, proto, idx + 1)
            roipooling3way_in2_layer_def = get_layer_def(net, proto, idx + 2)
            roipooling3way_out_layer_def = get_layer_def(net, proto, idx + 3)
            if func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                bottom0 = "*feat_maps.at(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "*feat_maps.at(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "*feat_maps.at(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                bottom0 = "feat_maps->Get(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "feat_maps->Get(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "feat_maps->Get(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            else:
                bottom0 = get_bottom(roipooling3way_in0_layer_def, blob_list_index, 0)
                bottom1 = get_bottom(roipooling3way_in1_layer_def, blob_list_index, 0)
                bottom2 = get_bottom(roipooling3way_in2_layer_def, blob_list_index, 0)
            top = get_top(roipooling3way_out_layer_def, blob_list_index, 0)

            function += "  cnn->roi_pooling_3way({}, {}, {}, rois, {}, {}, {}, {}, {}, {}{}, {});\n" \
                    .format(bottom0, bottom1, bottom2, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, layer_def.roi_pooling_param.spatial_scale / 2, layer_def.roi_pooling_param.spatial_scale / 4, top, fc2conv, roi_pooling_type[roipooling3way_in0_layer_def.roi_pooling_param.pool])

        elif is_roipooling3x3way(net.layers, idx):                
            roipooling3way_in0_layer_def = get_layer_def(net, proto, idx + 0)
            roipooling3way_in1_layer_def = get_layer_def(net, proto, idx + 1)
            roipooling3way_in2_layer_def = get_layer_def(net, proto, idx + 2)
            roipooling3way_in3_layer_def = get_layer_def(net, proto, idx + 4)
            roipooling3way_in6_layer_def = get_layer_def(net, proto, idx + 8)
            roipooling3way_out0_layer_def = get_layer_def(net, proto, idx + 3)
            roipooling3way_out1_layer_def = get_layer_def(net, proto, idx + 7)
            roipooling3way_out2_layer_def = get_layer_def(net, proto, idx + 11)
            if func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                bottom0 = "*feat_maps.at(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "*feat_maps.at(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "*feat_maps.at(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                bottom0 = "feat_maps->Get(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "feat_maps->Get(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "feat_maps->Get(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            else:
                bottom0 = get_bottom(roipooling3way_in0_layer_def, blob_list_index, 0)
                bottom1 = get_bottom(roipooling3way_in1_layer_def, blob_list_index, 0)
                bottom2 = get_bottom(roipooling3way_in2_layer_def, blob_list_index, 0)
            top0 = get_top(roipooling3way_out0_layer_def, blob_list_index, 0)
            top1 = get_top(roipooling3way_out1_layer_def, blob_list_index, 0)
            top2 = get_top(roipooling3way_out2_layer_def, blob_list_index, 0)

            function += "  cnn->roi_pooling_3x3way({}, {}, {}, rois, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}, {});\n" \
                    .format(bottom0, bottom1, bottom2, \
                    roipooling3way_in0_layer_def.roi_pooling_param.pooled_h, roipooling3way_in3_layer_def.roi_pooling_param.pooled_h, roipooling3way_in6_layer_def.roi_pooling_param.pooled_h, \
                    roipooling3way_in0_layer_def.roi_pooling_param.pooled_w, roipooling3way_in3_layer_def.roi_pooling_param.pooled_w, roipooling3way_in6_layer_def.roi_pooling_param.pooled_w, \
                    roipooling3way_in0_layer_def.roi_pooling_param.spatial_scale, roipooling3way_in1_layer_def.roi_pooling_param.spatial_scale, roipooling3way_in2_layer_def.roi_pooling_param.spatial_scale, \
                    top0, top1, top2, fc2conv, roi_pooling_type[roipooling3way_in0_layer_def.roi_pooling_param.pool])

        elif is_roipooling3x3way_multi_out(net.layers, idx):                
            roipooling3way_in0_layer_def = get_layer_def(net, proto, idx + 0)
            roipooling3way_in1_layer_def = get_layer_def(net, proto, idx + 1)
            roipooling3way_in2_layer_def = get_layer_def(net, proto, idx + 2)
            roipooling3way_in3_layer_def = get_layer_def(net, proto, idx + 4)
            roipooling3way_in6_layer_def = get_layer_def(net, proto, idx + 8)
            roipooling3way_out0_layer_def = get_layer_def(net, proto, idx + 12)
            roipooling3way_out1_layer_def = get_layer_def(net, proto, idx + 13)
            roipooling3way_out2_layer_def = get_layer_def(net, proto, idx + 14)
            if func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                bottom0 = "*feat_maps.at(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "*feat_maps.at(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "*feat_maps.at(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                bottom0 = "feat_maps->Get(\"{}\")".format(roipooling3way_in0_layer_def.bottom[0])
                bottom1 = "feat_maps->Get(\"{}\")".format(roipooling3way_in1_layer_def.bottom[0])
                bottom2 = "feat_maps->Get(\"{}\")".format(roipooling3way_in2_layer_def.bottom[0])
            else:
                bottom0 = get_bottom(roipooling3way_in0_layer_def, blob_list_index, 0)
                bottom1 = get_bottom(roipooling3way_in1_layer_def, blob_list_index, 0)
                bottom2 = get_bottom(roipooling3way_in2_layer_def, blob_list_index, 0)
            top0 += "std::vector<Tensor*> {"
            for t in xrange(len(roipooling3way_out0_layer_def.top)):
                top0 += "&{}, ".format(get_top(roipooling3way_out0_layer_def, blob_list_index, t))
            top0 = top0[:-2] + "}"
            top1 += "std::vector<Tensor*> {"
            for t in xrange(len(roipooling3way_out1_layer_def.top)):
                top1 += "&{}, ".format(get_top(roipooling3way_out1_layer_def, blob_list_index, t))
            top1 = top1[:-2] + "}"
            top2 += "std::vector<Tensor*> {"
            for t in xrange(len(roipooling3way_out2_layer_def.top)):
                top2 += "&{}, ".format(get_top(roipooling3way_out2_layer_def, blob_list_index, t))
            top2 = top2[:-2] + "}"
            function += "  cnn->roi_pooling_3x3way({}, {}, {}, rois, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}, {});\n" \
                    .format(bottom0, bottom1, bottom2, \
                    roipooling3way_in0_layer_def.roi_pooling_param.pooled_h, roipooling3way_in3_layer_def.roi_pooling_param.pooled_h, roipooling3way_in6_layer_def.roi_pooling_param.pooled_h, \
                    roipooling3way_in0_layer_def.roi_pooling_param.pooled_w, roipooling3way_in3_layer_def.roi_pooling_param.pooled_w, roipooling3way_in6_layer_def.roi_pooling_param.pooled_w, \
                    roipooling3way_in0_layer_def.roi_pooling_param.spatial_scale, roipooling3way_in1_layer_def.roi_pooling_param.spatial_scale, roipooling3way_in2_layer_def.roi_pooling_param.spatial_scale, \
                    top0, top1, top2, fc2conv, roi_pooling_type[roipooling3way_in0_layer_def.roi_pooling_param.pool])

        elif is_psroipooling(net.layers, idx):
            psroipooling3way_in0_layer_def = get_layer_def(net, proto, idx + 0)                
            psroipooling3way_in1_layer_def = get_layer_def(net, proto, idx + 3)
            psroipooling3way_out0_layer_def = get_layer_def(net, proto, idx + 2)
            psroipooling3way_out1_layer_def = get_layer_def(net, proto, idx + 5)
            if func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                bottom0 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in0_layer_def.bottom[0])
                bottom1 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in1_layer_def.bottom[0])
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                bottom0 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in0_layer_def.bottom[0])
                bottom1 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in1_layer_def.bottom[0])
            else:
                bottom0 = get_bottom(psroipooling3way_in0_layer_def, blob_list_index, 0)
                bottom1 = get_bottom(psroipooling3way_in1_layer_def, blob_list_index, 0)
            top0 = get_top(psroipooling3way_out0_layer_def, blob_list_index, 0)
            top1 = get_top(psroipooling3way_out1_layer_def, blob_list_index, 0)

            function += "  cnn->ps_roi_pooling({}, rois, {}, {}, {}, {});\n" \
                .format(bottom0, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, top0)
            function += "  cnn->ps_roi_pooling({}, rois, {}, {}, {}, {});\n" \
                .format(bottom1, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, top1)

        elif is_psroipooling3way(net.layers, idx):
            psroipooling3way_in0_layer_def = get_layer_def(net, proto, idx + 0)                
            psroipooling3way_in1_layer_def = get_layer_def(net, proto, idx + 3)
            psroipooling3way_in2_layer_def = get_layer_def(net, proto, idx + 6)                                                               
            psroipooling3way_in3_layer_def = get_layer_def(net, proto, idx + 9)
            psroipooling3way_in4_layer_def = get_layer_def(net, proto, idx + 12)
            psroipooling3way_in5_layer_def = get_layer_def(net, proto, idx + 15)
            psroipooling3way_out0_layer_def = get_layer_def(net, proto, idx + 18)
            psroipooling3way_out1_layer_def = get_layer_def(net, proto, idx + 19)

            if func == FUNC_TYPE.Detect_AD or func == FUNC_TYPE.Detect_MD or func == FUNC_TYPE.Recognize_FPR:
                bottom0 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in0_layer_def.bottom[0])
                bottom1 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in1_layer_def.bottom[0])
                bottom2 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in2_layer_def.bottom[0])
                bottom3 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in3_layer_def.bottom[0])
                bottom4 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in4_layer_def.bottom[0])
                bottom5 = "*feat_maps.at(\"{}\")".format(psroipooling3way_in5_layer_def.bottom[0])
            elif func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
                bottom0 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in0_layer_def.bottom[0])
                bottom1 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in1_layer_def.bottom[0])
                bottom2 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in2_layer_def.bottom[0])
                bottom3 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in3_layer_def.bottom[0])
                bottom4 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in4_layer_def.bottom[0])
                bottom5 = "feat_maps->Get(\"{}\")".format(psroipooling3way_in4_layer_def.bottom[0])
            else:
                bottom0 = get_bottom(psroipooling3way_in0_layer_def, blob_list_index, 0)
                bottom1 = get_bottom(psroipooling3way_in1_layer_def, blob_list_index, 0)
                bottom2 = get_bottom(psroipooling3way_in2_layer_def, blob_list_index, 0)
                bottom3 = get_bottom(psroipooling3way_in3_layer_def, blob_list_index, 0)
                bottom4 = get_bottom(psroipooling3way_in4_layer_def, blob_list_index, 0)
                bottom5 = get_bottom(psroipooling3way_in5_layer_def, blob_list_index, 0)
            top0 = get_top(psroipooling3way_out0_layer_def, blob_list_index, 0)
            top1 = get_top(psroipooling3way_out1_layer_def, blob_list_index, 0)

            function += "  cnn->ps_roi_pooling_3way({}, {}, {}, rois, {}, {}, {}, {}, {}, {});\n" \
                .format(bottom0, bottom2, bottom4, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, layer_def.roi_pooling_param.spatial_scale / 2, layer_def.roi_pooling_param.spatial_scale / 4, top0)
            function += "  cnn->ps_roi_pooling_3way({}, {}, {}, rois, {}, {}, {}, {}, {}, {});\n" \
                .format(bottom1, bottom3, bottom5, layer_def.roi_pooling_param.pooled_w, layer_def.roi_pooling_param.pooled_h, layer_def.roi_pooling_param.spatial_scale, layer_def.roi_pooling_param.spatial_scale / 2, layer_def.roi_pooling_param.spatial_scale / 4, top1)

        elif is_fc(net.layers, idx):
            if is_binary(net.layers, idx):
                function += "  cnn->binary_fc({}, layer_blobs[{}][0], layer_blobs[{}][1], layer_blobs[{}][2], {:d}, {}{});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_idx, layer_idx, layer_idx, layer_def.inner_product_param.bias_term, get_top(layer_def, blob_list_index, 0), fc2conv)
            else:
                function += "  cnn->fc({}, layer_blobs[{}][0], layer_blobs[{}][1], {:d}, {}{});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_idx, layer_idx, layer_def.inner_product_param.bias_term, get_top(layer_def, blob_list_index, 0), fc2conv)

        elif is_softmax(net.layers, idx):
            if layer_def.name == "line_seg_map" or layer_def.name == "ld_line_seg_map" or layer_def.name == "ld_seg_map":
                function += "#ifndef SPEEDUP_RPN\n"
                function += "  cnn->softmax({}, line_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#else\n"
                function += "  cnn->sigmoid({}, line_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#endif\n"
            elif layer_def.name == "boundary_seg_map":
                function += "#ifndef SPEEDUP_RPN\n"
                function += "  cnn->softmax({}, boundary_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#else\n"
                function += "  cnn->sigmoid({}, boundary_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#endif\n"
            elif layer_def.name == "vp_seg_map" or layer_def.name == "ld_vp_seg_map":
                function += "#ifndef SPEEDUP_RPN\n"
                function += "  cnn->softmax({}, vp_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#else\n"
                function += "  cnn->sigmoid({}, vp_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
                function += "#endif\n"
            elif layer_def.name=="clu2typeVal_prob" or layer_def.name == "ld_clu2typeVal_prob":
                function += "  cnn->softmax({}, line_type_val);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="clu2typePos_prob" or layer_def.name=="ld_clu2typePos_prob":
                function += "  cnn->softmax({}, line_type_pos);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="clu2typeShape_prob" or layer_def.name=="ld_clu2typeShape_prob":
                function += "  cnn->softmax({}, line_type_shape);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="clu2typeSD_prob" or layer_def.name=="ld_clu2typeSD_prob":
                function += "  cnn->softmax({}, line_type_sd);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="clu2typeColor_prob" or layer_def.name=="ld_clu2typeColor_prob":
                function += "  cnn->softmax({}, line_type_color);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="clu2typeBicycle_prob" or layer_def.name=="ld_clu2typeBicycle_prob":
                function += "  cnn->softmax({}, line_type_bicycle);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="boundary_clu2typeVal_prob":
                function += "  cnn->softmax({}, boundary_type_val);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="boundary_clu2typeShape_prob":
                function += "  cnn->softmax({}, boundary_type_shape);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name=="boundary_clu2typePos_prob":
                function += "  cnn->softmax({}, boundary_type_pos);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif (func == FUNC_TYPE.Segment_LBF or func == FUNC_TYPE.Segment_LF) and layer_def.name == "fsd_seg_prob":
                function += "  cnn->softmax({}, freespace_seg_map);\n" \
                .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name == "failsafe_out":
                function += "  cnn->softmax({}, {}{});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), "FS", fc2conv)
            elif layer_def.name == "vpy_out":
                function += "  cnn->softmax({}, {}{});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), "VL", fc2conv)
            elif layer_def.name == "closeness_out":
                function += "  cnn->softmax({}, {}{});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), "Close", fc2conv)
            else:
                function += "  cnn->softmax({}, {}{});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_top(layer_def, blob_list_index, 0), fc2conv)
        elif is_sigmoid(net.layers, idx):
            if layer_def.name == "line_seg_map" or layer_def.name == "ld_seg_map":
                function += "  cnn->sigmoid({}, line_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name == "vp_seg_map" or layer_def.name == "ld_vp_seg_map":
                function += "  cnn->sigmoid({}, vp_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
            elif layer_def.name == "boundary_seg_map":
                function += "  cnn->sigmoid({}, boundary_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
            elif (func == FUNC_TYPE.Segment_LBF or func == FUNC_TYPE.Segment_LF) and layer_def.name == "fsd_seg_prob":
                function += "  cnn->sigmoid({}, freespace_seg_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0))
            else :
                function += "  cnn->sigmoid({}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), get_top(layer_def, blob_list_index, 0))
                

        elif is_detection(net.layers, idx):
            if len(layer_def.bottom) == 4 and len(layer_def.detection_param.rpn_option) == 0:
                bottom0 = get_bottom(layer_def, blob_list_index, 1)
                bottom1 = get_bottom(layer_def, blob_list_index, 0)
            else:
                bottom0 = "std::vector<const Tensor*> {"
                for b in xrange(len(layer_def.bottom[:-1])):
                    if b % 3 == 1:
                        bottom0 += "&{}, ".format(get_bottom(layer_def, blob_list_index, b))
                bottom0 = bottom0[:-2] + "}"
                bottom1 = "std::vector<const Tensor*> {"
                for b in xrange(len(layer_def.bottom[:-1])):
                    if b % 3 == 0:
                        bottom1 += "&{}, ".format(get_bottom(layer_def, blob_list_index, b))
                bottom1 = bottom1[:-2] + "}"
            
            
            if len(layer_def.detection_param.rpn_option) == 0:
                if func == FUNC_TYPE.BRNetProc:
                    function += "  if (!feat_maps->GetTiling()) {\n"
                    function += "    cnn->get_det_output(rois, {}, {}, feat_maps->GetImageWidth(), feat_maps->GetImageHeight(), feat_maps->GetDetCropX(), feat_maps->GetDetCropY(), feat_maps->GetDetResizedWidth(), feat_maps->GetDetResizedHeight(), feat_maps->GetDetScaleX(), feat_maps->GetDetScaleY(), {}, {}, {}, {}, {}, {}, {}, {}, detection_boxes, class_map, true{});\n" \
                        .format(bottom0, bottom1, layer_def.detection_param.mean_dx, layer_def.detection_param.mean_dy, layer_def.detection_param.mean_dw, layer_def.detection_param.mean_dh, layer_def.detection_param.std_dx, layer_def.detection_param.std_dy, layer_def.detection_param.std_dw, layer_def.detection_param.std_dh, fc2conv)
                    function += "  } else {\n"
                    function += "    cnn->get_det_output_tiles(rois, {}, {}, feat_maps->GetImageWidth(), feat_maps->GetImageHeight(), feat_maps->GetDetCropXs(), feat_maps->GetDetCropYs(), feat_maps->GetDetResizedXs(), feat_maps->GetDetResizedWidths(), feat_maps->GetDetResizedHeights(), feat_maps->GetDetScaleXs(), feat_maps->GetDetScaleYs(), {}, {}, {}, {}, {}, {}, {}, {}, detection_boxes, class_map, true{});\n" \
                        .format(bottom0, bottom1, layer_def.detection_param.mean_dx, layer_def.detection_param.mean_dy, layer_def.detection_param.mean_dw, layer_def.detection_param.mean_dh, layer_def.detection_param.std_dx, layer_def.detection_param.std_dy, layer_def.detection_param.std_dw, layer_def.detection_param.std_dh, fc2conv)
                    function += "  }\n"
                else:
                    function += "  if (!enabled_tiling) {\n"
                    function += "    cnn->get_det_output(rois, {}, {}, image_width, image_height, det_crop_x, det_crop_y, det_resized_width, det_resized_height, det_scale_x, det_scale_y, {}, {}, {}, {}, {}, {}, {}, {}, options.NMS_THRES, options.DETECTION_THRES1, options.DETECTION_THRES2, options.REFINE_THRES1, options.VOTE_THRES, restore_scale, refine_size, objects, class_map, options.SUCC_SUPPRESSED_CNT, options.NMS_VOTE{});\n" \
                        .format(bottom0, bottom1, layer_def.detection_param.mean_dx, layer_def.detection_param.mean_dy, layer_def.detection_param.mean_dw, layer_def.detection_param.mean_dh, layer_def.detection_param.std_dx, layer_def.detection_param.std_dy, layer_def.detection_param.std_dw, layer_def.detection_param.std_dh, fc2conv)
                    function += "  } else {\n"
                    function += "    cnn->get_det_output_tiles(rois, {}, {}, image_width, image_height, det_crop_xs, det_crop_ys, det_resized_xs, det_resized_widths, det_resized_heights, det_scale_xs, det_scale_ys, {}, {}, {}, {}, {}, {}, {}, {}, options.NMS_THRES, options.NMS_TILE_THRES1, options.NMS_TILE_THRES2, options.DETECTION_THRES1, options.DETECTION_THRES2, options.REFINE_THRES1, options.VOTE_THRES, restore_scale, refine_size, objects, class_map, options.SUCC_SUPPRESSED_CNT, options.NMS_VOTE{});\n" \
                        .format(bottom0, bottom1, layer_def.detection_param.mean_dx, layer_def.detection_param.mean_dy, layer_def.detection_param.mean_dw, layer_def.detection_param.mean_dh, layer_def.detection_param.std_dx, layer_def.detection_param.std_dy, layer_def.detection_param.std_dw, layer_def.detection_param.std_dh, fc2conv)
                    function += "  }\n"
            else:
                anchors = ""
                feat_stride = ""
                base_size = ""
                min_size = ""
                max_size = ""
                for i in xrange(len(layer_def.detection_param.rpn_option)):
                    anchors += "&anchors[{}], ".format(i)
                    feat_stride += "{}, ".format(layer_def.detection_param.rpn_option[i].feat_stride)
                    base_size += "{}, ".format(layer_def.detection_param.rpn_option[i].base_size)
                    min_size += "{}, ".format(layer_def.detection_param.rpn_option[i].min_size)
                    max_size += "{}, ".format(layer_def.detection_param.rpn_option[i].max_size)
                anchors = "{" + anchors[:-2] + "}"
                feat_stride = "{" + feat_stride[:-2] + "}"
                base_size = "{" + base_size[:-2] + "}"
                min_size = "{" + min_size[:-2] + "}"
                max_size = "{" + max_size[:-2] + "}"            
                function += "  const std::vector<const Tensor*> detection_anchors = {};\n".format(anchors)
                function += "  const std::vector<int32_t> detection_feat_stride = {};\n".format(feat_stride)
                function += "  const std::vector<int32_t> detection_base_size = {};\n".format(base_size)
                function += "  const std::vector<int32_t> detection_min_size = {};\n".format(min_size)
                function += "  const std::vector<int32_t> detection_max_size = {};\n".format(max_size)
                if func == FUNC_TYPE.BRNetProc:
                    function += "  if (!feat_maps->GetTiling()) {\n"
                    function += "    cnn->get_det_output(rois, {}, {}, detection_anchors, detection_feat_stride, detection_base_size, detection_min_size, detection_max_size, feat_maps->GetImageWidth(), feat_maps->GetImageHeight(), feat_maps->GetDetCropX(), feat_maps->GetDetCropY(), feat_maps->GetDetResizedWidth(), feat_maps->GetDetResizedHeight(), feat_maps->GetDetScaleX(), feat_maps->GetDetScaleY(), detection_boxes, class_map, true);\n" \
                        .format(bottom0, bottom1)
                    function += "  } else {\n"
                    function += "    cnn->get_det_output_tiles(rois, {}, {}, detection_anchors, detection_feat_stride, detection_base_size, detection_min_size, detection_max_size, feat_maps->GetImageWidth(), feat_maps->GetImageHeight(), feat_maps->GetDetCropXs(), feat_maps->GetDetCropYs(), feat_maps->GetDetResizedXs(), feat_maps->GetDetResizedWidths(), feat_maps->GetDetResizedHeights(), feat_maps->GetDetScaleXs(), feat_maps->GetDetScaleYs(), detection_boxes, class_map, true);\n" \
                        .format(bottom0, bottom1)
                    function += "  }\n"
                else:
                    function += "  if (!enabled_tiling) {\n"
                    function += "    cnn->get_det_output(rois, {}, {}, detection_anchors, detection_feat_stride, detection_base_size, detection_min_size, detection_max_size, image_width, image_height, det_crop_x, det_crop_y, det_resized_width, det_resized_height, det_scale_x, det_scale_y, options.NMS_THRES, options.DETECTION_THRES1, options.DETECTION_THRES2, options.REFINE_THRES1, options.VOTE_THRES, restore_scale, refine_size, objects, class_map, options.SUCC_SUPPRESSED_CNT, options.NMS_VOTE);\n" \
                        .format(bottom0, bottom1)
                    function += "  } else {\n"
                    function += "    cnn->get_det_output_tiles(rois, {}, {}, detection_anchors, detection_feat_stride, detection_base_size, detection_min_size, detection_max_size, image_width, image_height, det_crop_xs, det_crop_ys, det_resized_xs, det_resized_widths, det_resized_heights, det_scale_xs, det_scale_ys, options.NMS_THRES, options.NMS_TILE_THRES1, options.NMS_TILE_THRES2, options.DETECTION_THRES1, options.DETECTION_THRES2, options.REFINE_THRES1, options.VOTE_THRES, restore_scale, refine_size, objects, class_map, options.SUCC_SUPPRESSED_CNT, options.NMS_VOTE);\n" \
                        .format(bottom0, bottom1)
                    function += "  }\n"

        elif is_laplacian(net.layers, idx):
            laplacian_pool_layer_def = get_layer_def(net, proto, idx + 2)
            laplacian_mul_layer_def = get_layer_def(net, proto, idx + 4)
            laplacian_out_layer_def = get_layer_def(net, proto, idx + 5)

            function += "  cnn->laplacian_mask({}, {}, {}, {}, {}, {}, {}, {});\n" \
                .format(get_bottom(laplacian_mul_layer_def, blob_list_index, 0), get_bottom(laplacian_out_layer_def, blob_list_index, 0), get_bottom(layer_def, blob_list_index, 0), laplacian_pool_layer_def.pooling_param.kernel_size, laplacian_pool_layer_def.pooling_param.kernel_size, laplacian_pool_layer_def.pooling_param.stride, laplacian_pool_layer_def.pooling_param.pad, get_top(laplacian_out_layer_def, blob_list_index, 0))            

        elif is_segmentation(net.layers, idx):
            if start_layer == start_layers[FUNC_TYPE.Segment_LF] or start_layer == start_layers[FUNC_TYPE.Segment_LBF]:
                function = function.replace(get_bottom(layer_def, blob_list_index, 0), "freespace_seg_map")
            else:
                function = function.replace(get_bottom(layer_def, blob_list_index, 0), "seg_map")

        elif is_curvefitting(net.layers, idx):
            function = function.replace(get_bottom(layer_def, blob_list_index, 1), "*ld_maps[0]")
            function = function.replace(get_bottom(layer_def, blob_list_index, 0), "*ld_maps[1]")

        elif is_detection_3d(net.layers, idx) or is_post_detection(net.layers, idx):
            sources = "std::vector<const Tensor*> {&rois, "
            source = "&cnn->to_cpu({}), " if func == FUNC_TYPE.ADNetProc else "&{}, "
            sources += source.format(get_bottom(layer_def, blob_list_index, 1))
            source = "&cnn->to_cpu({}, true), " if func == FUNC_TYPE.ADNetProc else "&{}, "
            sources += source.format(get_bottom(layer_def, blob_list_index, 0))
            if (len(layer_def.bottom) == 4 and layer_def.post_detection_param.num_shape == 2 and layer_def.post_detection_param.num_pts == 4):
                post_type = "POST_3D"
                attribute_group = "NULL"
                regression_group = "NULL"
            elif ((len(layer_def.bottom) == 4 and layer_def.post_detection_param.num_shape == 1 and (layer_def.post_detection_param.num_pts == 8 or layer_def.post_detection_param.num_pts == 16)) or len(layer_def.bottom) == 5):
                if len(layer_def.bottom) == 5:
                    sources += source.format(get_bottom(layer_def, blob_list_index, 2))
                post_type = "POST_NEW_3D"
                attribute_group = "NULL"
                regression_group = "NULL"
            elif ((len(layer_def.bottom) == 8 and layer_def.post_detection_param.num_shape == 1 and (layer_def.post_detection_param.num_pts == 8 or layer_def.post_detection_param.num_pts == 16)) or len(layer_def.bottom) == 9):
                bdix = 2
                post_type = "POST_V2"
                if len(layer_def.bottom) == 9:
                    sources += source.format(get_bottom(layer_def, blob_list_index, bdix))
                    bdix += 1
                    if layer_def.post_detection_param.num_shape == 1 and layer_def.post_detection_param.num_pts == 16:
                        post_type = "POST_V4"
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix))
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix + 1))
                function += "  const int attribute_group[] = {{ {} }};\n".format(layer_def.post_detection_param.attribute_group[1:-1].strip())
                function += "  const int regression_group[] = {{ {} }};\n".format(layer_def.post_detection_param.regression_group[1:-1].strip())
                attribute_group = "attribute_group"
                regression_group = "regression_group"
            elif (len(layer_def.bottom) == 8):
                sources += source.format(get_bottom(layer_def, blob_list_index, 2))
                sources += source.format(get_bottom(layer_def, blob_list_index, 3))
                post_type = "POST_V1"
                function += "  const int attribute_group[] = {{ {} }};\n".format(layer_def.post_detection_param.attribute_group[1:-1].strip())
                function += "  const int regression_group[] = {{ {} }};\n".format(layer_def.post_detection_param.regression_group[1:-1].strip())
                attribute_group = "attribute_group"
                regression_group = "regression_group"
            elif ((len(layer_def.bottom) == 11 and layer_def.post_detection_param.num_shape == 1 and (layer_def.post_detection_param.num_pts == 8 or layer_def.post_detection_param.num_pts == 16)) or len(layer_def.bottom) == 12):
                bdix = 2
                post_type = "POST_V3"
                if len(layer_def.bottom) == 12:
                    sources += source.format(get_bottom(layer_def, blob_list_index, bdix))
                    bdix += 1
                    if layer_def.post_detection_param.num_shape == 1 and layer_def.post_detection_param.num_pts == 16:
                        post_type = "POST_V4"
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix))
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix + 1))
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix + 2))
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix + 3))
                sources += source.format(get_bottom(layer_def, blob_list_index, bdix + 4))
                if func != FUNC_TYPE.ADNetProc:
                    function += "  const int attribute_group[] = {{ {} }};\n".format(layer_def.post_detection_param.attribute_group[1:-1].strip())
                    function += "  const int regression_group[] = {{ {} }};\n".format(layer_def.post_detection_param.regression_group[1:-1].strip())
                    attribute_group = "attribute_group"
                    regression_group = "regression_group"
            sources = sources[:-2] + "}"

            if func == FUNC_TYPE.ADNetProc:
                function += "  this->get_output(feat_maps, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, true, attributes{});\n" \
                    .format(sources, post_type, layer_def.detection_3d_param.mean_dx, layer_def.detection_3d_param.mean_dy, layer_def.detection_3d_param.mean_dw, layer_def.detection_3d_param.mean_dh, layer_def.detection_3d_param.std_dx, layer_def.detection_3d_param.std_dy, layer_def.detection_3d_param.std_dw, layer_def.detection_3d_param.std_dh, fc2conv)
            else:
                function += "  if (!object_detector_impl->GetTiling()) {\n"                    
                function += "    cnn->get_det_attribute_output({}, {}, object_detector_impl->GetImageWidth(), object_detector_impl->GetImageHeight(), object_detector_impl->GetDetCropX(), object_detector_impl->GetDetCropY(), object_detector_impl->GetDetResizedWidth(), object_detector_impl->GetDetResizedHeight(), object_detector_impl->GetDetScaleX(), object_detector_impl->GetDetScaleY(), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, restore_scale, attributes{});\n" \
                    .format(sources, post_type, layer_def.detection_3d_param.mean_dx, layer_def.detection_3d_param.mean_dy, layer_def.detection_3d_param.mean_dw, layer_def.detection_3d_param.mean_dh, layer_def.detection_3d_param.std_dx, layer_def.detection_3d_param.std_dy, layer_def.detection_3d_param.std_dw, layer_def.detection_3d_param.std_dh, attribute_group, regression_group, fc2conv)
                function += "  } else {\n"
                function += "    cnn->get_det_attribute_output_tiles({}, {}, object_detector_impl->GetImageWidth(), object_detector_impl->GetImageHeight(), object_detector_impl->GetDetCropXs(), object_detector_impl->GetDetCropYs(), object_detector_impl->GetDetResizedXs(), object_detector_impl->GetDetResizedWidths(), object_detector_impl->GetDetResizedHeights(), object_detector_impl->GetDetScaleXs(), object_detector_impl->GetDetScaleYs(), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, restore_scale, attributes{});\n" \
                    .format(sources, post_type, layer_def.detection_3d_param.mean_dx, layer_def.detection_3d_param.mean_dy, layer_def.detection_3d_param.mean_dw, layer_def.detection_3d_param.mean_dh, layer_def.detection_3d_param.std_dx, layer_def.detection_3d_param.std_dy, layer_def.detection_3d_param.std_dw, layer_def.detection_3d_param.std_dh, attribute_group, regression_group, fc2conv)
                function += "  }\n"

        elif is_space2depth(net.layers, idx):
            function += "  cnn->space2depth({}, {}, {});\n" \
                .format(get_bottom(layer_def, blob_list_index, 0), layer_def.space2depth_param.scale, get_top(layer_def, blob_list_index, 0))

        elif is_depth2space(net.layers, idx):
            if layer_def.name == "cluster_score_d2s" or layer_def.name == "ld_cluster_score_d2s":
                function += "  cnn->depth2space({}, {}, line_cluster_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.depth2space_param.scale)
            elif layer_def.name == "boundary_cluster_score_d2s":
                function += "  cnn->depth2space({}, {}, boundary_cluster_map);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.depth2space_param.scale)
            else:
                function += "  cnn->depth2space({}, {}, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 0), layer_def.depth2space_param.scale, get_top(layer_def, blob_list_index, 0))

        elif is_quantized_mask_pooling(net.layers, idx):
            if layer_def.name=='boundary_quantized_feature1':
                function += '#ifndef SPEEDUP_RPN\n'
                function += "  cnn->quantized_mask_pooling(boundary_cluster_map, std::vector<const Tensor*> {{&{}}}, boundary_seg_map, boundary_cluster_min_val, boundary_cluster_max_val, 256, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(layer_def, blob_list_index, 0))
                function += '#else\n'
                function += "  cnn->quantized_mask_pooling(boundary_cluster_map, std::vector<const Tensor*> {{&{}}}, {}, boundary_cluster_min_val, boundary_cluster_max_val, 256, {}, false);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(get_layer_def_by_name(proto, "boundary_seg_map"), blob_list_index, 0), get_top(layer_def, blob_list_index, 0))
                function += '#endif\n'
            elif layer_def.name=='ld_quantized_feature1':
                function += '#ifndef SPEEDUP_RPN\n'
                function += "  cnn->quantized_mask_pooling(line_cluster_map, std::vector<const Tensor*> {{&{}}}, line_seg_map, cluster_min_val, cluster_max_val, 256, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(layer_def, blob_list_index, 0))
                function += '#else\n'
                function += "  cnn->quantized_mask_pooling(line_cluster_map, std::vector<const Tensor*> {{&{}}}, {}, cluster_min_val, cluster_max_val, 256, {}, false);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(get_layer_def_by_name(proto, "ld_seg_map"), blob_list_index, 0), get_top(layer_def, blob_list_index, 0))
                function += '#endif\n'
            else:
                function += '#ifndef SPEEDUP_RPN\n'
                function += "  cnn->quantized_mask_pooling(line_cluster_map, std::vector<const Tensor*> {{&{}}}, line_seg_map, cluster_min_val, cluster_max_val, 256, {});\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(layer_def, blob_list_index, 0))
                function += '#else\n'
                function += "  cnn->quantized_mask_pooling(line_cluster_map, std::vector<const Tensor*> {{&{}}}, {}, cluster_min_val, cluster_max_val, 256, {}, false);\n" \
                    .format(get_bottom(layer_def, blob_list_index, 2), get_top(get_layer_def_by_name(proto, "line_seg_map"), blob_list_index, 0), get_top(layer_def, blob_list_index, 0))
                function += '#endif\n'

        elif is_split(net.layers, idx):
            for top_idx, top in enumerate(layer_def.top):       
                blob_list_index[layer_def.top[top_idx]] = blob_list_index[layer_def.bottom[0]]
        else:
            unknown_layer = True
        
        if unknown_layer:
            function = function[:-len("  // {}\n".format(layer_def.name))]
        else:
            function += "  EVT_CHECK(\"{}::{}::{}\");\n".format(net_name, func_name, layer_def.name)

        if is_disable_layer(layer_def, net, disable_layers_for_speedup_rpn):
            function += "#endif\n"

        if layer_def.name in end_layers[FUNC_TYPE.Detect_MD] or layer_def.name in end_layers[FUNC_TYPE.MDNetProc]:
            # for maskrcnn
            function += "  // mask_prob\n"
            function += "  cnn->sigmoid({}, {});\n".format(get_top(layer_def, blob_list_index, 0), get_top(layer_def, blob_list_index, 0))
            function += "  EVT_CHECK(\"{}::{}::{}\");\n".format(net_name, func_name, "mask_prob")
            function += "  // mask_detection\n"
            if func == FUNC_TYPE.MDNetProc:
                function += "  this->get_output(cnn->to_cpu({}, true), masks);\n".format(get_top(layer_def, blob_list_index, 0))
            else:
                function += "  this->get_output({}, masks);\n".format(get_top(layer_def, blob_list_index, 0))
            function += "  EVT_CHECK(\"{}::{}::{}\");\n".format(net_name, func_name, "mask_detection")
            end_idx = idx
        elif layer_def.name in end_layers[FUNC_TYPE.Recognize_FPR]:
            # for false positive recognizer
            function += "  // fp_recognition\n"
            function += "  cnn->get_false_positive_output({}, false_positives, options.FPR_THRES);\n".format(get_top(layer_def, blob_list_index, 0))
            function += "  EVT_CHECK(\"{}::{}::{}\");\n".format(net_name, func_name, "fp_recognition")
            end_idx = idx
        elif layer_def.name in end_layer:
            end_idx = idx

        if end_idx != -1:
            break

    last_evt_check = function.rfind("EVT_CHECK")
    function = function[:last_evt_check] + "EVT_END" + function[last_evt_check + len("EVT_CHECK"):]
    if function.count("\n") == 1:
        function = ""
    else:
        if func == FUNC_TYPE.CFNetProc or func == FUNC_TYPE.RPNetProc or func == FUNC_TYPE.BRNetProc or func == FUNC_TYPE.ADNetProc or func == FUNC_TYPE.MDNetProc:
            function += "  return SV_SUCCESS;\n"
        function += "}\n\n"

    return function

def make_defines(net_name, is_int8):
    defines = ""
    defines += "#ifdef SPEEDUP_RPN\n"
    defines += "#define FOLD true\n"
    defines += "#else // SPEEDUP_RPN\n"
    defines += "#define FOLD false\n"
    defines += "#endif // SPEEDUP_RPN\n\n"

    defines += "#ifdef SPEEDUP_FC2CONV\n"
    defines += "#define FC2CONV true\n"
    defines += "#else // SPEEDUP_FC2CONV\n"
    defines += "#define FC2CONV false\n"
    defines += "#endif // SPEEDUP_FC2CONV\n\n"

    if is_int8:
        defines += "#ifdef USE_INT\n"
        defines += "#define LOAD_INT_W(idx, shift_bit, max_bit) layer_blobs[idx][0].LoadInt({}_int8_wt, offset_int8, (INT_BIT_W - 8) + shift_bit, (INT_BIT_W - 8) + (max_bit))\n".format(net_name)
        defines += "#define LOAD_INT_B(idx, shift_bit, max_bit) layer_blobs[idx][1].LoadInt({}_int8_wt, offset_int8, (INT_BIT_B - 8) + shift_bit, (INT_BIT_B - 8) + (max_bit))\n".format(net_name)
        defines += "#define LOAD_INT_W_FOLD(idx, shift_bit, max_bit) layer_blobs[idx][0].LoadInt({}_int8_wt, offset_int8, (INT_BIT_W - 8) + shift_bit, (INT_BIT_W - 8) + (max_bit), FOLD)\n".format(net_name)
        defines += "#define LOAD_INT_B_FOLD(idx, shift_bit, max_bit) layer_blobs[idx][1].LoadInt({}_int8_wt, offset_int8, (INT_BIT_B - 8) + shift_bit, (INT_BIT_B - 8) + (max_bit), FOLD)\n".format(net_name)            
        defines += "#define SET_SHIFTN_W(idx, shift_bit) layer_blobs[idx][0].shift_n = (INT_BIT_W - 8) + (shift_bit)\n"
        defines += "#define SET_SHIFTN_B(idx, shift_bit) layer_blobs[idx][1].shift_n = (INT_BIT_B - 8) + (shift_bit)\n"
        defines += "#define SET_SHIFTN_D(idx, shift_bit) data_blobs[idx].shift_n = (INT_BIT_B - 8) + (shift_bit)\n"
        defines += "#else\n"
        defines += "#define LOAD_INT_W(idx, shift_bit, max_bit)\n"
        defines += "#define LOAD_INT_B(idx, shift_bit, max_bit)\n"
        defines += "#define LOAD_INT_W_FOLD(idx, shift_bit, max_bit)\n"
        defines += "#define LOAD_INT_B_FOLD(idx, shift_bit, max_bit)\n"
        defines += "#define SET_SHIFTN_W(idx, shift_bit)\n"
        defines += "#define SET_SHIFTN_B(idx, shift_bit)\n"
        defines += "#define SET_SHIFTN_D(idx, shift_bit)\n"
        defines += "#endif\n\n"
    return defines

def make_class_names_function(proto, net_name):
    class_names = parse_param_str(proto.class_names)
    class_names_function = ""
    class_names_function += "int32_t {}::GetNumClasses() const {{\n".format(net_name)
    class_names_function += "  return {};\n".format(len(class_names))
    class_names_function += "}\n\n"
    class_names_function += "const char_t* {}::ClassName(int32_t index) const {{\n".format(net_name)
    class_names_function += "  static const char_t class_names[{}][{}] = {{ ".format(len(class_names), max([len(class_name) for class_name in class_names]) + 1)
    for idx, class_name in enumerate(class_names):
        class_names_function += "\"{}\"".format(class_name)
        if idx < len(class_names) - 1:
            class_names_function += ", "
    class_names_function += " };\n"
    class_names_function += "  return class_names[index];\n"
    class_names_function += "}\n\n"
    class_names_function += "IObjectDetector::ClassType {}::ClassType(int32_t index) const {{\n".format(net_name)
    class_names_function += "  static const IObjectDetector::ClassType class_types[{}] = {{\n".format(len(class_names))
    for idx, class_name in enumerate(class_names):
        class_names_function += "    {}".format(class_type_map[class_name])
        if idx < len(class_names) - 1:
            class_names_function += ",\n"
    class_names_function += "\n  };\n"
    class_names_function += "  return class_types[index];\n"
    class_names_function += "}\n\n"
    return class_names_function

def make_class_names_function_for_tsr(proto, net_name, region):
  
    region_prefix = ''
    if region.lower() in ['korea', 'kor']:
      region_prefix = 'k_'
    elif region.lower() in ['germany', 'ger']:
      region_prefix = 'g_'
    elif region.lower() in ['japan', 'jp']:
      region_prefix = 'j_'

    class_names = parse_param_str(proto.class_names)
    class_names_function = ""
    class_names_function += "int32_t {}::GetNumClasses() const {{\n".format(net_name)
    class_names_function += "  return {};\n".format(len(class_names))
    class_names_function += "}\n\n"
    class_names_function += "const char_t* {}::ClassName(int32_t index) const {{\n".format(net_name)
    class_names_function += "  static const char_t class_names[{}][{}] = {{ ".format(len(class_names), max([len(region_prefix + class_name) for class_name in class_names]) + 1)
    for idx, class_name in enumerate(class_names):
        class_names_function += "\"{}\"".format(region_prefix + class_name)
        if idx < len(class_names) - 1:
            class_names_function += ", "
    class_names_function += " };\n"
    class_names_function += "  return class_names[index];\n"
    class_names_function += "}\n\n"

    return class_names_function

def make_set_default_option_function(proto, net_name):
    set_default_option_function = ""    
    if len(proto.class_thresholds) > 0:
        det_thresh1 = [ 0.5 if th > 0.5 else th for th in parse_param_str(proto.class_thresholds)]
        det_thresh2 = parse_param_str(proto.class_thresholds)
        str_det_thresh1 = str(det_thresh1).replace("'", "")
        str_det_thresh2 = str(det_thresh2).replace("'", "")
        set_default_option_function += "void {}::SetDefaultOptions() {{\n".format(net_name)
        set_default_option_function += "  options.NUM_DET_CLASSES = GetNumClasses();\n"
        set_default_option_function += "  options.DETECTION_THRES1.resize(options.NUM_DET_CLASSES);\n"
        set_default_option_function += "  options.DETECTION_THRES2.resize(options.NUM_DET_CLASSES);\n"
        set_default_option_function += "  options.SetOption(\"DETECTION_THRES1\", \"{}\");\n".format(str_det_thresh1)
        set_default_option_function += "  options.SetOption(\"DETECTION_THRES2\", \"{}\");\n".format(str_det_thresh2)
        set_default_option_function += "}\n\n"
    return set_default_option_function

def load_caffe(args):
    caffe.set_mode_cpu()
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    proto = caffe_pb2.NetParameter()
    file = open(args.prototxt, "r")
    google.protobuf.text_format.Merge(file.read(), proto)
    file.close()
    
    is_int8 = args.prototxt_int8 is not None and args.caffemodel_int8 is not None
    net_int8 = None
    proto_int8 = None
    if is_int8:
        net_int8 = caffe.Net(args.prototxt_int8, args.caffemodel_int8, caffe.TEST)
        proto_int8 = caffe_pb2.NetParameter()
        file_int8 = open(args.prototxt_int8, "r")
        google.protobuf.text_format.Merge(file_int8.read(), proto_int8)
        file_int8.close()

    net_name = args.name
    if net_name is None: net_name = proto.name
    if net_name == "": net_name = "AnonymousNet"
    net_name = net_name.strip().replace(" ", "_")
        
    
    is_int8_qt = args.quantize_info is not None
    quantize_info = None
    if is_int8_qt:
        with open(args.quantize_info, "rb") as f:
            quantize_info = cPickle.load(f) 

    return net, proto, net_name, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info

def make_vsproject(net_name, weights_file_list, out, skip_weights):
    if not skip_weights:
        sample_vcxproj = '<?xml version="1.0" encoding="utf-8"?>\n<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n  <ItemGroup Label="ProjectConfigurations">\n    <ProjectConfiguration Include="Debug|x64">\n      <Configuration>Debug</Configuration>\n      <Platform>x64</Platform>\n    </ProjectConfiguration>\n    <ProjectConfiguration Include="Release|x64">\n      <Configuration>Release</Configuration>\n      <Platform>x64</Platform>\n    </ProjectConfiguration>\n  </ItemGroup>\n  <ItemGroup>\nReplace1\n  </ItemGroup>\n  <PropertyGroup Label="Globals">\n    <ProjectGuid>{Replace2}</ProjectGuid>\n    <Keyword>Win32Proj</Keyword>\n    <RootNamespace>libsvneth</RootNamespace>\n    <WindowsTargetPlatformVersion>8.1</WindowsTargetPlatformVersion>\n  </PropertyGroup>\n  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />\n  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'" Label="Configuration">\n    <ConfigurationType>StaticLibrary</ConfigurationType>\n    <UseDebugLibraries>true</UseDebugLibraries>\n    <PlatformToolset>v140</PlatformToolset>\n    <CharacterSet>Unicode</CharacterSet>\n  </PropertyGroup>\n  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'" Label="Configuration">\n    <ConfigurationType>StaticLibrary</ConfigurationType>\n    <UseDebugLibraries>false</UseDebugLibraries>\n    <PlatformToolset>v140</PlatformToolset>\n    <WholeProgramOptimization>true</WholeProgramOptimization>\n    <CharacterSet>Unicode</CharacterSet>\n  </PropertyGroup>\n  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />\n  <ImportGroup Label="ExtensionSettings">\n  </ImportGroup>\n  <ImportGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'" Label="PropertySheets">\n    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists(\'$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\')" Label="LocalAppDataPlatform" />\n  </ImportGroup>\n  <ImportGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'" Label="PropertySheets">\n    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists(\'$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\')" Label="LocalAppDataPlatform" />\n  </ImportGroup>\n  <PropertyGroup Label="UserMacros" />\n  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">\n    <OutDir>..\\build\\svnet\\$(Platform)\\$(Configuration)\\bin\\</OutDir>\n    <IntDir>..\\build\\$(ProjectName)\\$(Platform)\\$(Configuration)\\obj\\</IntDir>\n  </PropertyGroup>\n  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">\n    <OutDir>..\\build\\svnet\\$(Platform)\\$(Configuration)\\bin\\</OutDir>\n    <IntDir>..\\build\\$(ProjectName)\\$(Platform)\\$(Configuration)\\obj\\</IntDir>\n  </PropertyGroup>\n  <ItemDefinitionGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">\n    <ClCompile>\n      <PrecompiledHeader>\n      </PrecompiledHeader>\n      <WarningLevel>Level3</WarningLevel>\n      <Optimization>\n      </Optimization>\n      <PreprocessorDefinitions>Replace3;WIN32;_DEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n    </ClCompile>\n    <Link>\n      <SubSystem>Windows</SubSystem>\n      <GenerateDebugInformation>true</GenerateDebugInformation>\n    </Link>\n  </ItemDefinitionGroup>\n  <ItemDefinitionGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">\n    <ClCompile>\n      <WarningLevel>Level3</WarningLevel>\n      <PrecompiledHeader>\n      </PrecompiledHeader>\n      <Optimization>\n      </Optimization>\n      <FunctionLevelLinking>true</FunctionLevelLinking>\n      <IntrinsicFunctions>true</IntrinsicFunctions>\n      <PreprocessorDefinitions>Replace3;WIN32;NDEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n    </ClCompile>\n    <Link>\n      <SubSystem>Windows</SubSystem>\n      <GenerateDebugInformation>true</GenerateDebugInformation>\n      <EnableCOMDATFolding>true</EnableCOMDATFolding>\n      <OptimizeReferences>true</OptimizeReferences>\n    </Link>\n  </ItemDefinitionGroup>\n  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />\n  <ImportGroup Label="ExtensionTargets">\n  </ImportGroup>\n</Project>'
        sample_vcxproj_filters = '<?xml version="1.0" encoding="utf-8"?>\n<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n  <ItemGroup>\nReplace1\n  </ItemGroup>\n</Project>'

        replace1_txt = ""
        for filename in weights_file_list:
            replace1_txt += "    <ClCompile Include=\"" + filename + "\" />\n"
        replace1_txt = replace1_txt[:-1]

        alltxt = sample_vcxproj
        alltxt = alltxt.replace("Replace1", replace1_txt)
        alltxt = alltxt.replace("Replace2", str(uuid.uuid4()))
        alltxt = alltxt.replace("Replace3", net_name.upper())
        f = open(os.path.join(out, "weights", "{}.vcxproj".format(net_name.lower())), "w")
        f.write(alltxt)
        f.close()

        alltxt = sample_vcxproj_filters
        alltxt = alltxt.replace("Replace1", replace1_txt)
        f = open(os.path.join(out, "weights", "{}.vcxproj.filters".format(net_name.lower())), "w")
        f.write(alltxt)
        f.close()    

def build_lib(net_name, weights_file_list, out, skip_weights, vs_ver="14.1"):
    if skip_weights == 1:
        return
    if vs_ver == "14.0":
        set_path = 'set PATH=C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\amd64'
    elif vs_ver == "14.1":
        set_path = 'set PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Tools\\MSVC\\14.16.27023\\bin\\HostX86\\x64;C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Tools\\MSVC\\14.16.27023\\bin\\HostX86\\x86;C:\\Program Files (x86)\\Windows Kits\\8.1\\bin\\x86;C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\Common7\\tools;C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\Common7\\ide;C:\\Program Files (x86)\\HTML Help Workshop;C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\MSBuild\\15.0\\Bin;C:\\WINDOWS\\Microsoft.NET\\Framework\\v4.0.30319\\;'
    else:
        assert "not supported VS"
    cl_cmd = 'cl.exe /GS /GL /W3 /Gy /Zc:wchar_t /Zi /Gm- /Zc:inline /fp:precise /D "{}" /D "WIN32" /D "NDEBUG" /D "_LIB" /D "_UNICODE" /D "UNICODE" /errorReport:prompt /WX- /Zc:forScope /Gd /Oi /MD /FC /EHsc /nologo -c'.format(net_name.upper())
    for filename in weights_file_list:
        cl_cmd += " " + os.path.join(out, "weights", filename)
    lib_cmd = 'lib.exe /LTCG /MACHINE:X64 /NOLOGO /out:{}.lib'.format(os.path.join(out, "weights", net_name.lower()))
    for filename in weights_file_list:
        lib_cmd += " " + filename[:-3] + 'obj'
    del_cmd = "del "
    for filename in weights_file_list:
        del_cmd += " " + filename[:-3] + 'obj'
    os.system(set_path + "&" + cl_cmd)
    os.system(set_path + "&" + lib_cmd)
    os.system(del_cmd)
