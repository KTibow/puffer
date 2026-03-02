#include "roomba.h"

#define Env Roomba
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->width = unpack(kwargs, "width");
  env->height = unpack(kwargs, "height");
  env->speed = unpack(kwargs, "speed");
  env->dt = unpack(kwargs, "dt");
  env->tick_limit = unpack(kwargs, "tick_limit");
  env->wheel_base = unpack(kwargs, "wheel_base");
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  return 0;
}
