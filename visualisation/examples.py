from NormSystem import NormSystem, get_generic_h_scale, get_generic_dim_scale, change_color

o_scale = get_generic_dim_scale(dim=NormSystem.dim_o, num_entries=9)
r_scale = get_generic_dim_scale(dim=NormSystem.dim_r, num_entries=4)
s_scale = get_generic_dim_scale(dim=NormSystem.dim_s, num_entries=7)
t_scale = get_generic_dim_scale(dim=NormSystem.dim_t, num_entries=5)
h_scale = get_generic_h_scale(num_levels=3)


def outside(norm_system):
    """
    Example for cases not subsumed by a norm. The the three closest norms are displayed. First in a simple example with
    only one norm and one case, then extended to two cases and the original norm and then further extended to the two
    cases with three norms.

    Creates figures for each constellation for one dimension and two dimensions in fully detail and a slightly less
    detailed form.

    :param norm_system: The normsystem to display the cases and the norms in.
    """
    # one case
    saveid = 'outside_simple'
    norm_system.add_case(o_val=o_scale[8], r_val=r_scale[3], s_val=s_scale[3], t_val=t_scale[4])
    norm_system.add_norm(o_vals=(o_scale[2], o_scale[5]), r_vals=(r_scale[0], r_scale[1]),
                         s_vals=(s_scale[4], s_scale[5]), t_vals=(t_scale[3], t_scale[3]),
                         hierarchy=h_scale[0], starttime='99', norm_type=NormSystem.type_prohib,
                         identifier='\\alpha')
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=True)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=True)

    # two cases
    saveid = 'outside_simple_two_cases'
    norm_system.add_case(o_val=o_scale[0], r_val=r_scale[3], s_val=s_scale[0], t_val=t_scale[0])
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=True)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=True)

    # with more norms
    saveid = 'outside_complex'
    norm_system.add_norm(o_vals=(o_scale[0], o_scale[8]), r_vals=(r_scale[0], r_scale[1]),
                         s_vals=(s_scale[0], s_scale[2]), t_vals=(t_scale[0], t_scale[0]),
                         hierarchy=h_scale[0], starttime='18', norm_type=NormSystem.type_obl,
                         identifier='\\beta')
    norm_system.add_norm(o_vals=(o_scale[6], o_scale[8]), r_vals=(r_scale[3], r_scale[3]),
                         s_vals=(s_scale[0], s_scale[0]), t_vals=(t_scale[1], t_scale[2]),
                         hierarchy=h_scale[0], starttime='01', norm_type=NormSystem.type_perm,
                         identifier='\\gamma')
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r), detailed=False,
                                  saveidentifier=saveid)
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r), detailed=True,
                                  saveidentifier=saveid)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=True)


def subsumption(norm_system):
    """
    Example for a subsumption. Even though three norms are given, only two are displayed in the resulting figure as
    these norms subsume the given case. If at least one norm subsumes a case, all other norms not subsuming the case
    are considered irrelevant to the case and are not displayed.

    Creates figures for one dimension and two dimensions in fully detail and a slightly less detailed form.

    :param norm_system: The normsystem to display the case and the norms in.
    """
    norm_system.add_norm(o_vals=(o_scale[1], o_scale[5]), r_vals=(r_scale[1], r_scale[2]),
                         s_vals=(s_scale[2], s_scale[3]), t_vals=(t_scale[3], t_scale[4]),
                         hierarchy=h_scale[0], starttime='79', norm_type=NormSystem.type_obl,
                         identifier='\\alpha')
    norm_system.add_norm(o_vals=(o_scale[3], o_scale[5]), r_vals=(r_scale[1], r_scale[3]),
                         s_vals=(s_scale[5], s_scale[6]), t_vals=(t_scale[0], t_scale[1]),
                         hierarchy=h_scale[1], starttime='85', norm_type=NormSystem.type_obl,
                         identifier='\\beta')
    norm_system.add_norm(o_vals=(o_scale[3], o_scale[4]), r_vals=(r_scale[2], r_scale[3]),
                         s_vals=(s_scale[0], s_scale[2]), t_vals=(t_scale[2], t_scale[3]),
                         hierarchy=h_scale[2], starttime='90', norm_type=NormSystem.type_prohib,
                         identifier='\\gamma')
    norm_system.add_case(o_val=o_scale[3], r_val=r_scale[2], s_val=s_scale[2], t_val=t_scale[3])

    saveid = 'subsumption'
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                  dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                  saveidentifier=saveid, detailed=True)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=False)
    norm_system.draw_dims_one_all(saveidentifier=saveid, detailed=True)


if __name__ == "__main__":
    main_norm_system = NormSystem(object_scale=o_scale, r_scale=r_scale, subject_scale=s_scale,
                                  time_scale=t_scale, hierarchy_scale=h_scale)
    subsumption(norm_system=main_norm_system)
    main_norm_system.reset()
    change_color(NormSystem.type_obl, '#a0978d')
    outside(norm_system=main_norm_system)
    print('Done with examples')
