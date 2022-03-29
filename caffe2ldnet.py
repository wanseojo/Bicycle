#!/usr/bin/env python
import caffe2svnet3
import sys, os

def make_ldnet(net, proto, net_name, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info, out, skip_weights):
    impl_head = "{}::{}()".format(net_name, net_name)
    concrete_head  = "{}_PLATFORM::{}_PLATFORM()".format(net_name, net_name)
    concrete_body  = "  cnn = CNNTools_PLATFORM::GetInstance();\n"
    concrete_body += "  image2tensor_ = Image2Tensor_PLATFORM::GetInstance();\n"
    concrete_body += "  Initialize();\n"
    defines = caffe2svnet3.make_defines(net_name, is_int8)
    constructor, wt_def, weights_file_list, blob_list, blob_list_index, layer_list, layer_list_index = \
        caffe2svnet3.make_constructor_and_weights(net, proto, net_name, caffe2svnet3.start_layers[caffe2svnet3.FUNC_TYPE.Segment_LD], caffe2svnet3.end_layers[caffe2svnet3.FUNC_TYPE.Segment_LD], impl_head, concrete_head, concrete_body, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info, out, skip_weights, use_ifdef=False)    
    functions  = caffe2svnet3.make_function(net, proto, net_name, caffe2svnet3.FUNC_TYPE.Segment_LD, blob_list, blob_list_index, layer_list, layer_list_index, 1)
    defines = replace_for_ldnet(defines)
    constructor = replace_for_ldnet(constructor)
    functions = replace_for_ldnet(functions)
    type_classes_info = "void {}::SetTypeClassesInfo()".format(net_name)
    type_classes_info += "{\n"
    type_classes_info = set_typeclases(type_classes_info)
    type_classes_info += "}\n"
    f = open(os.path.join(out, "src", "nets", "{}.cpp".format(net_name.lower())), "w")
    f.write("#if defined({}) || defined(SVNETALL)\n".format(net_name.upper()))
    f.write("#include \"{}.hpp\"\n".format(net_name.lower()))
    f.write("#include \"prof.hpp\"\n")
    f.write("#ifdef USE_FP16\n")
    f.write("#include <fp16.h>\n")
    f.write("#endif\n\n")
    f.write(defines)
    f.write("#define REGISTER_NET REGISTER_LD_NET\n\n")
    f.write("namespace sv {\n")
    f.write(wt_def)
    f.write(constructor)
    f.write(type_classes_info)
    f.write(functions)
    f.write("} // namespace sv\n")
    f.write("#endif\n")
    f.close()

    concrete_definition  = "class {}_PLATFORM : public {} {{\n".format(net_name, net_name)
    concrete_definition += " public:\n"
    concrete_definition += "  {}_PLATFORM();\n".format(net_name)
    concrete_definition += "  virtual const char* GetType() {{ return \"{}_PLATFORM\"; }}\n".format(net_name)
    concrete_definition += "  virtual bool UseGPU() { return false; }\n"
    concrete_definition += "  virtual OperationType GetOperationType() { return OP_PLATFORM; }\n"
    concrete_definition += "};\n"

    f = open(os.path.join(out, "src", "nets", "{}.hpp".format(net_name.lower())), "w")
    f.write("#ifndef _{}_HPP_\n".format(net_name.upper()))
    f.write("#define _{}_HPP_\n".format(net_name.upper()))
    f.write("#ifdef _MSC_VER\n")
    f.write("#pragma comment (lib, \"{}.lib\")\n".format(net_name.lower()))
    f.write("#endif\n\n")
    f.write("#include \"lanenet/lanenet.hpp\"\n\n")
    f.write("namespace sv {\n")
    f.write("class {} : public LaneNet {{\n".format(net_name))
    f.write(" public:\n")
    f.write("  {}();\n".format(net_name))
    f.write(" protected:\n")
    f.write("  virtual void {};\n".format(caffe2svnet3.functions[caffe2svnet3.FUNC_TYPE.Segment_LD])) 
    f.write("  virtual void SetTypeClassesInfo();\n")
    f.write("};\n\n")
    f.write(concrete_definition.replace("PLATFORM", "CPU"))
    f.write("#ifdef USE_INT\n")
    f.write(concrete_definition.replace("PLATFORM", "INT"))
    f.write("#endif // USE_INT\n")
    f.write("#ifdef USE_GPU\n")
    f.write(concrete_definition.replace("PLATFORM", "GPU").replace("false", "true"))
    f.write("#ifdef USE_FP16\n")
    f.write(concrete_definition.replace("PLATFORM", "FP16").replace("false", "true"))
    f.write("#endif // USE_FP16\n")
    f.write("#endif // USE_GPU \n")
    f.write("} // namespace sv\n")
    f.write("#endif // _{}_HPP_\n".format(net_name.upper()))
    f.close()

    return weights_file_list

def replace_for_ldnet(s):
    s = s.replace("layer_blobs", "layer_blobs_")
    s = s.replace("data_blobs", "data_blobs_")
    s = s.replace("cnn", "cnn_")
    s = s.replace("image_width = image.width", "image_width_ = image.width")
    s = s.replace("image_height = image.height", "image_height_ = image.height")
    return s

def set_typeclases(type_classes_info):
    type_classes_info += "  // lane typeVal\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[0].class_num = 2;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[0].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_LINE_VALIDITY_Invalid;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[0].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_LINE_VALIDITY_Valid;\n\n"
    type_classes_info += "  // lane typeShape\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[1].class_num = 3;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[1].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_PATTERN_Solid;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[1].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_PATTERN_Dashed;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[1].class_id[2] = LaneBoundaryClassID::LDA_CLASS_ID_PATTERN_BottsDot;\n\n"
    type_classes_info += "  // lane typeSD\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[2].class_num = 2;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[2].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_MULTIPLE_LINE_Single;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[2].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_MULTIPLE_LINE_Double;\n\n"
    type_classes_info += "  // lane typePos\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_num = 13;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_LeftOpposite;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_BRANCH;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[2] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborLeftLeftLeft;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[3] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborLeftLeft;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[4] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborLeft;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[5] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_EgoLeft;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[6] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_EgoRight;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[7] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborRight;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[8] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborRightRight;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[9] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_NeighborRightRightRight;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[10] = LaneBoundaryClassID::LDA_CLASS_ID_MERGED;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[11] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_RightOpposite;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[3].class_id[12] = LaneBoundaryClassID::LDA_CLASS_ID_POSITION_Unknown;\n\n"
    type_classes_info += "  // lane typeColor\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[4].class_num = 4;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[4].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_COLOR_White;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[4].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_COLOR_Yellow;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[4].class_id[2] = LaneBoundaryClassID::LDA_CLASS_ID_COLOR_Blue;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[4].class_id[3] = LaneBoundaryClassID::LDA_CLASS_ID_COLOR_Unknown;\n\n"
    type_classes_info += "  // lane typeBicycle\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[5].class_num = 2;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[5].class_id[0] = LaneBoundaryClassID::LDA_CLASS_ID_BICYCLE_IS;\n"
    type_classes_info += "  lane_type_maps_.typemap_infos[5].class_id[1] = LaneBoundaryClassID::LDA_CLASS_ID_BICYCLE_NOT;\n"
    return type_classes_info

if __name__ == "__main__":
    #sys.argv += "--def ../../../svcaffe/models/train/Lane/L3255/L3255_SV190322_MV190607_SKT190322_FT190410x2_HM190522_mult4_train_val.prototxt --net ../../../svcaffe/weights/train/Lane/Lane_L3255_iter_2200000.caffemodel --def_int8 ../../../svcaffe/models/train/Lane/L3255/L3255_SV190322_MV190607_SKT190322_FT190410x2_HM190522_mult4_train_val_INT8.prototxt --net_int8 ../../../svcaffe/weights/train/Lane/Lane_L3255_iter_2200000_INT8.caffemodel --quantize_info ../../../svcaffe/weights/train/Lane/Lane_L3255_iter_2200000_INT8.pkl --out ../../ --name LDNetL3255 --skip_weights 0".split(" ")

    args = caffe2svnet3.parse_args()

    net, proto, net_name, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info = caffe2svnet3.load_caffe(args)    
    weights_file_list = make_ldnet(net, proto, net_name, is_int8, net_int8, proto_int8, is_int8_qt, quantize_info, args.out, args.skip_weights)
    caffe2svnet3.build_lib(net_name, weights_file_list, args.out, args.skip_weights)
