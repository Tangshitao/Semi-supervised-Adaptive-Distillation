# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Top-level control of the building/installation/cleaning of various targets

NVCC := nvcc
MAXAS := maxas.pl
INSTALL_MAXAS := yes
MAXAS_URL := https://github.com/NervanaSystems/maxas.git

PKG_NAME := nervanagpu
TEST_FILES := convnet-benchmarks.py

KERNEL_DIR := $(PKG_NAME)/kernels
CUSRC_DIR := $(KERNEL_DIR)/cu
CUASS_DIR := $(KERNEL_DIR)/sass
CUBIN_DIR := $(KERNEL_DIR)/cubin
DOC_DIR := doc
PY_DEPS = setup.py requirements.txt $(wildcard $(PKG_NAME)/*.py)

# these control test options and attribute filters
NOSE_FLAGS := ""  # --pdb --pdb-failures
NOSE_ATTRS := -a '!slow'

define op_template
  # expected parameters:
  # 1: operation class (gemm, conv, pool)
  # 2: width prefix (h, s)
  # 3: operation type/order (_fprop, _bprop, _updat, _nn, _max, ...)
  # 4: operation code variable (, _, _vec_, _s8_, _u8_, ...)
  # 5: operation size (, 128x128, K64_N64, C128_K128, ...)
  # We set target specific flags and add to opclass_OPS only when the
  # combination of parameters exists as a named .cu file
  ifneq ("$(wildcard $(CUSRC_DIR)/$(2)$(1)$(3)$(4)$(5).cu)","")
    $(1)_OPS += $(addprefix $(CUBIN_DIR)/, \
                  $(addsuffix .cubin,$(2)$(1)$(3)$(4)$(5)))
    ifneq ($(filter $(3),_bprop _updat _max),)
      ifeq ($(2),h)
        NVCCFLAGS_$(2)$(1)$(3)$(4)$(5) := -arch sm_52
      else
        NVCCFLAGS_$(2)$(1)$(3)$(4)$(5) := -arch sm_50
      endif
    else
      NVCCFLAGS_$(2)$(1)$(3)$(4)$(5) := -arch sm_50
    endif
    MAXASFLAGS_$(2)$(1)$(3)$(4)$(5) := -i
    ifneq ($(strip $(subst _,,$(4))),)
      MAXASFLAGS_$(2)$(1)$(3)$(4)$(5) += -k $(2)$(1)$(3)$(4)$(5) \
                                         -D$(subst _,,$(4)) 1
    endif
  endif
endef

define strip_codes
  $(subst _vec,,$(subst _s8,,$(subst _u8,,$(1))))
endef

define list_includes
  $(shell sed -rn 's/^<INCLUDE file="(.*)"\/>/\1/p' $(call strip_codes,$(1)))
endef

WIDTHS := h s # h == half (16bit), s == single (32bit)

GEMM_ORDERS := _nn _nt _tn
GEMM_CODES := _ _vec_
GEMM_SIZES := 128x128 128x64 128x32 128x16 32x128
$(foreach w,$(WIDTHS), \
  $(foreach o,$(GEMM_ORDERS), \
    $(foreach c,$(GEMM_CODES), \
      $(foreach s,$(GEMM_SIZES), \
        $(eval $(call op_template,gemm,$(w),$(o),$(c),$(s))) \
      )  \
    ) \
  ) \
)

POOL_TYPES := _max
$(foreach w,$(WIDTHS), \
  $(foreach t,$(POOL_TYPES), \
    $(eval $(call op_template,pool,$(w),$(t))) \
  ) \
)

CONV_TYPES := _fprop _bprop _updat
CONV_CODES := _ # _s8_ _u8_
CONV_SIZES := C32_N64 C64_N64 K64_N64 C64_K64 C128_K64 C128_K128
$(foreach w,$(WIDTHS), \
  $(foreach t,$(CONV_TYPES), \
    $(foreach c,$(CONV_CODES), \
      $(foreach s,$(CONV_SIZES), \
        $(eval $(call op_template,conv,$(w),$(t),$(c),$(s))) \
      ) \
    ) \
  ) \
)


.PHONY: all kernels maxas_check python install doc html test clean uninstall

all: kernels python

kernels: maxas_check $(CUBIN_DIR) $(gemm_OPS) $(pool_OPS) $(conv_OPS)

maxas_check:
ifeq (, $(shell which $(MAXAS)))
  ifeq ($(INSTALL_MAXAS), yes)
		@echo "installing maxas..."
		@tmpdir=`mktemp -d -t nervanagpu.XXXXXXXX` \
			|| { echo "failed to create temp file"; exit 1; } ;\
		echo $$tmpdir &&\
		cd $$tmpdir &&\
		git clone $(MAXAS_URL) &&\
		cd $(basename $(MAXAS)) &&\
		perl Makefile.PL &&\
		make install ;\
		if [ $$? != 0 ] ; then \
			rm -rf $$tmpdir ;\
			echo "problems installing maxas"; exit 1 ;\
		else \
			rm -rf $$tmpdir ;\
		fi
  else
		$(error "$(MAXAS) not found.  See: $(MAXAS_URL)")
  endif
endif

$(CUBIN_DIR):
	@mkdir -p $(CUBIN_DIR)

$(PY_DEPS):
	@echo "updating: $(PY_DEPS)"
	@touch $@

.SECONDEXPANSION:
$(CUBIN_DIR)/%.cubin: $(CUSRC_DIR)/%.cu \
	                    $$(call strip_codes,$(CUASS_DIR)/%.sass) \
	                    $$(call list_includes,$(CUASS_DIR)/%.sass)
	@echo "building kernel: $*..."
	@$(NVCC) $(NVCCFLAGS_$*) -cubin $< -o $@
	@$(MAXAS) $(MAXASFLAGS_$*) $(call strip_codes,$(CUASS_DIR)/$*.sass) $@

.python_install_required: $(PY_DEPS) $(wildcard $(CUBIN_DIR)/*.cubin)
ifneq (, $(shell pip show $(PKG_NAME)))
	@echo "removing existing $(PKG_NAME) python bindings..."
	@pip uninstall -y $(PKG_NAME)
endif
	@echo "installing $(PKG_NAME) python bindings..."
	@pip install .
	@touch $@

python: .python_install_required

install: python

doc:
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html

html: doc

test: kernels
	@echo "Running unit tests..."
	nosetests $(NOSE_ATTRS) $(NOSE_FLAGS) nervanagpu

bench: python
	@for t in $(TEST_FILES) ; do \
		echo "Running $$t..." ; \
		python benchmarks/$$t ; \
	done

clean:
	@rm -rf $(CUBIN_DIR)

uninstall:
	@echo "uninstalling python bindings..."
	@pip uninstall $(PKG_NAME)
