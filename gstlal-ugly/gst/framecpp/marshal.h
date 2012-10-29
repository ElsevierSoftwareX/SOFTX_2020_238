#ifndef __FRAMECPP_MARSHAL_H__
#define __FRAMECPP_MARSHAL_H__

#include	<glib-object.h>

G_BEGIN_DECLS

extern void framecpp_marshal_FLOW_RETURN__CLOCK_TIME__CLOCK_TIME(
	GClosure     *closure,
	GValue       *return_value,
	guint         n_param_values,
	const GValue *param_values,
	gpointer      invocation_hint,
	gpointer      marshal_data
);

extern void framecpp_marshal_VOID__CLOCK_TIME__CLOCK_TIME(
	GClosure     *closure,
	GValue       *return_value,
	guint         n_param_values,
	const GValue *param_values,
	gpointer      invocation_hint,
	gpointer      marshal_data
);

G_END_DECLS

#endif /* __FRAMECPP_MARSHAL_H__ */
