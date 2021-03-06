# Copyright (c) 2013 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# GNU Makefile based on shared rules provided by the Native Client SDK.
# See README.Makefiles for more details.

VALID_TOOLCHAINS := pnacl newlib glibc linux

NACL_SDK_ROOT ?= $(abspath $(CURDIR)/../../..)

TARGET = media_stream_video


include $(NACL_SDK_ROOT)/tools/common.mk


LIBS = ppapi_gles2 ppapi_cpp ppapi pthread \
	opencv_contrib opencv_stitching	opencv_nonfree opencv_superres opencv_ts opencv_videostab \
	opencv_gpu opencv_photo	opencv_objdetect opencv_legacy opencv_video opencv_ml opencv_calib3d \
	opencv_features2d opencv_highgui opencv_imgproc opencv_flann opencv_core png jpeg z m 


CFLAGS = -Wall
SOURCES = media_stream_video.cc 
#url_loader_handler.cc file_handler.cc

# Build rules generated by macros from common.mk:

$(foreach src,$(SOURCES),$(eval $(call COMPILE_RULE,$(src),$(CFLAGS))))

# The PNaCl workflow uses both an unstripped and finalized/stripped binary.
# On NaCl, only produce a stripped binary for Release configs (not Debug).
ifneq (,$(or $(findstring pnacl,$(TOOLCHAIN)),$(findstring Release,$(CONFIG))))
$(eval $(call LINK_RULE,$(TARGET)_unstripped,$(SOURCES),$(LIBS),$(DEPS)))
$(eval $(call STRIP_RULE,$(TARGET),$(TARGET)_unstripped))
else
$(eval $(call LINK_RULE,$(TARGET),$(SOURCES),$(LIBS),$(DEPS)))
endif

$(eval $(call NMF_RULE,$(TARGET),))
