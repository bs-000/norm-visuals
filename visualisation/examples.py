from NormSystem import NormSystem, get_generic_h_scale, get_generic_dim_scale, display_types, CaseAspectValues, \
    DimValue, NormAspectValues, Norm

o_scale = get_generic_dim_scale(dim=NormSystem.dim_o, num_entries=9)
r_scale = get_generic_dim_scale(dim=NormSystem.dim_r, num_entries=4)
s_scale = get_generic_dim_scale(dim=NormSystem.dim_s, num_entries=7)
t_scale = get_generic_dim_scale(dim=NormSystem.dim_t, num_entries=5)
h_scale = get_generic_h_scale(num_levels=3)


def get_dimval(scale, value):
    return DimValue(name=scale[value], value=value)

def save_figures(norm_system, saveid, aspect_name):
    norm_system.draw_dims_three(saveidentifier=saveid, dim_to_exclude=NormSystem.dim_o,
                                aspect_name=aspect_name)
    for display_t in display_types:
        norm_system.draw_dims_one_all(saveidentifier=saveid, display_type=display_t,
                                      aspect_name=aspect_name)
        norm_system.draw_dims_two_all(dims_one=(NormSystem.dim_o, NormSystem.dim_s),
                                      dims_two=(NormSystem.dim_t, NormSystem.dim_r),
                                      saveidentifier=saveid, display_type=display_t,
                                      aspect_name=aspect_name)


def outside(norm_system):
    """
    Example for cases not subsumed by a norm_aspect_value. The three closest norms are displayed. First in a simple example with
    only one norm_aspect_value and one case, then extended to two cases and the original norm_aspect_value and then further extended to the two
    cases with three norms.

    Creates figures for each constellation for one dimension and two dimensions in fully detail and a slightly less
    detailed form.

    :param norm_system: The normsystem to display the cases and the norms in.
    """
    # one case
    saveid = 'outside_simple'
    norm_system.add_case(domain_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[8], value=8),
                         r_val=DimValue(name=r_scale[3], value=3),
                         s_val=DimValue(name=s_scale[3], value=3),
                         t_val=DimValue(name=t_scale[4], value=4),
                         aspect_name=NormSystem.aspect_domain),
                         premise_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[8], value=8),
                                          r_val=DimValue(name=r_scale[3], value=3),
                                          s_val=DimValue(name=s_scale[3], value=3),
                                          t_val=DimValue(name=t_scale[4], value=4),
                                          aspect_name=NormSystem.aspect_premise)
                         )

    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[2], value=2),
                                     DimValue(name=o_scale[5], value=5)),
                             r_vals=(DimValue(name=r_scale[0], value=0),
                                     DimValue(name=r_scale[1], value=1)),
                             s_vals=(DimValue(name=s_scale[4], value=4),
                                     DimValue(name=s_scale[5], value=5)),
                             t_vals=(DimValue(name=t_scale[3], value=3),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_prohib,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='99', identifier='\\alpha',
                             aspect_name=NormSystem.aspect_domain),
                         ))
    norm_system.add_norm_premise_aspect(norm_identifier='\\alpha',
                             premise_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[2], value=2),
                                     DimValue(name=o_scale[5], value=5)),
                             r_vals=(DimValue(name=r_scale[0], value=0),
                                     DimValue(name=r_scale[1], value=1)),
                             s_vals=(DimValue(name=s_scale[4], value=4),
                                     DimValue(name=s_scale[5], value=5)),
                             t_vals=(DimValue(name=t_scale[3], value=3),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_dero_pos,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='99', identifier='\\alpha',
                             aspect_name=NormSystem.aspect_premise))
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_premise)
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_domain)


    # two cases
    saveid = 'outside_simple_two_cases'
    norm_system.add_case(domain_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[0], value=0),
                         r_val=DimValue(name=r_scale[3], value=3),
                         s_val=DimValue(name=s_scale[0], value=0),
                         t_val=DimValue(name=t_scale[0], value=0),
                         aspect_name=NormSystem.aspect_domain),
                         premise_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[0], value=0),
                         r_val=DimValue(name=r_scale[3], value=3),
                         s_val=DimValue(name=s_scale[0], value=0),
                         t_val=DimValue(name=t_scale[0], value=0),
                                          aspect_name=NormSystem.aspect_premise)
                         )
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_premise)
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_domain)


    # with more norms
    saveid = 'outside_complex'
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[0], value=0),
                                     DimValue(name=o_scale[8], value=8)),
                             r_vals=(DimValue(name=r_scale[1], value=1),
                                     DimValue(name=r_scale[1], value=1)),
                             s_vals=(DimValue(name=s_scale[0], value=0),
                                     DimValue(name=s_scale[2], value=2)),
                             t_vals=(DimValue(name=t_scale[0], value=0),
                                     DimValue(name=t_scale[0], value=0)),
                             norm_type=NormSystem.type_obl,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='18', identifier='\\beta',
                             aspect_name=NormSystem.aspect_domain))
                         )
    norm_system.add_norm_premise_aspect(norm_identifier='\\beta',
                                        premise_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[2], value=2),
                                     DimValue(name=o_scale[5], value=5)),
                             r_vals=(DimValue(name=r_scale[0], value=0),
                                     DimValue(name=r_scale[1], value=1)),
                             s_vals=(DimValue(name=s_scale[4], value=4),
                                     DimValue(name=s_scale[5], value=5)),
                             t_vals=(DimValue(name=t_scale[3], value=3),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_dero_pos,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='18', identifier='\\beta',
                             aspect_name=NormSystem.aspect_premise)
)
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[6], value=6),
                                     DimValue(name=o_scale[8], value=8)),
                             r_vals=(DimValue(name=r_scale[3], value=3),
                                     DimValue(name=r_scale[3], value=3)),
                             s_vals=(DimValue(name=s_scale[0], value=0),
                                     DimValue(name=s_scale[0], value=0)),
                             t_vals=(DimValue(name=t_scale[2], value=2),
                                     DimValue(name=t_scale[2], value=2)),
                             norm_type=NormSystem.type_perm,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='01', identifier='\\gamma',
                             aspect_name=NormSystem.aspect_domain),))
    norm_system.add_norm_premise_aspect(norm_identifier='\\gamma',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[1], value=1)),
                                            s_vals=(DimValue(name=s_scale[4], value=4),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_dero_pos,
                                            h_val=DimValue(name=h_scale[0], value=0),
                                            starttime='01', identifier='\\gamma',
                                            aspect_name=NormSystem.aspect_premise)
                                        )
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_premise)
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_domain)



def subsumption(norm_system):
    """
    Example for a subsumption. Even though three norms are given, only two are displayed in the resulting figure as
    these norms subsume the given case. If at least one norm_aspect_value subsumes a case, all other norms not subsuming the case
    are considered irrelevant to the case and are not displayed.

    Creates figures for one dimension and two dimensions in fully detail and a slightly less detailed form.

    :param norm_system: The normsystem to display the case and the norms in.
    """
    norm_system.add_case(domain_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[3], value=3),
                         r_val=DimValue(name=r_scale[2], value=2),
                         s_val=DimValue(name=s_scale[2], value=2),
                         t_val=DimValue(name=t_scale[3], value=3),
                         aspect_name=NormSystem.aspect_domain),
                         premise_aspect=
                         CaseAspectValues(o_val=DimValue(name=o_scale[3], value=3),
                         r_val=DimValue(name=r_scale[2], value=2),
                         s_val=DimValue(name=s_scale[2], value=2),
                         t_val=DimValue(name=t_scale[3], value=3),
                                          aspect_name=NormSystem.aspect_premise))

    # subsumes perfectly
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[1], value=1),
                                     DimValue(name=o_scale[5], value=5)),
                             r_vals=(DimValue(name=r_scale[1], value=1),
                                     DimValue(name=r_scale[2], value=2)),
                             s_vals=(DimValue(name=s_scale[2], value=2),
                                     DimValue(name=s_scale[3], value=3)),
                             t_vals=(DimValue(name=t_scale[3], value=3),
                                     DimValue(name=t_scale[4], value=4)),
                             norm_type=NormSystem.type_obl,
                             h_val=DimValue(name=h_scale[0], value=0),
                             starttime='79', identifier='\\alpha',
                             aspect_name=NormSystem.aspect_domain),) )
    norm_system.add_norm_premise_aspect(norm_identifier='\\alpha',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[2], value=2)),
                                            s_vals=(DimValue(name=s_scale[2], value=2),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_empow_pos,
                                            h_val=DimValue(name=h_scale[0], value=0),
                                            starttime='01', identifier='\\alpha',
                                            aspect_name=NormSystem.aspect_premise))
    # does not subsume in domain: ViVa zeigen, ViNo nicht zeigen
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[3], value=3),
                                     DimValue(name=o_scale[5], value=5)),
                             r_vals=(DimValue(name=r_scale[1], value=1),
                                     DimValue(name=r_scale[3], value=3)),
                             s_vals=(DimValue(name=s_scale[5], value=5),
                                     DimValue(name=s_scale[6], value=6)),
                             t_vals=(DimValue(name=t_scale[0], value=0),
                                     DimValue(name=t_scale[1], value=1)),
                             norm_type=NormSystem.type_obl,
                             h_val=DimValue(name=h_scale[1], value=1),
                             starttime='85', identifier='\\beta',
                             aspect_name=NormSystem.aspect_domain),))
    norm_system.add_norm_premise_aspect(norm_identifier='\\beta',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[2], value=2)),
                                            s_vals=(DimValue(name=s_scale[2], value=2),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_empow_pos,
                                            h_val=DimValue(name=h_scale[1], value=1),
                                            starttime='85', identifier='\\beta',
                                            aspect_name=NormSystem.aspect_premise))
    # does not subsume in premise: ViVa zeigen, Vino nicht
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[3], value=3),
                                     DimValue(name=o_scale[4], value=4)),
                             r_vals=(DimValue(name=r_scale[2], value=2),
                                     DimValue(name=r_scale[3], value=3)),
                             s_vals=(DimValue(name=s_scale[0], value=0),
                                     DimValue(name=s_scale[2], value=2)),
                             t_vals=(DimValue(name=t_scale[2], value=2),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_prohib,
                             h_val=DimValue(name=h_scale[2], value=2),
                             starttime='90', identifier='\\gamma',
                             aspect_name=NormSystem.aspect_domain),))
    norm_system.add_norm_premise_aspect(norm_identifier='\\gamma',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[1], value=1)),
                                            s_vals=(DimValue(name=s_scale[4], value=4),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_empow_pos,
                                            h_val=DimValue(name=h_scale[2], value=2),
                                            starttime='90', identifier='\\gamma',
                                            aspect_name=NormSystem.aspect_premise))

    # derogated in premise: ViVa aber nicht ViNo
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[3], value=3),
                                     DimValue(name=o_scale[4], value=4)),
                             r_vals=(DimValue(name=r_scale[2], value=2),
                                     DimValue(name=r_scale[3], value=3)),
                             s_vals=(DimValue(name=s_scale[0], value=0),
                                     DimValue(name=s_scale[2], value=2)),
                             t_vals=(DimValue(name=t_scale[2], value=2),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_prohib,
                             h_val=DimValue(name=h_scale[2], value=2),
                             starttime='90', identifier='\\delta',
                             aspect_name=NormSystem.aspect_domain),))
    norm_system.add_norm_premise_aspect(norm_identifier='\\delta',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[2], value=2)),
                                            s_vals=(DimValue(name=s_scale[2], value=2),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_dero_pos,
                                            h_val=DimValue(name=h_scale[2], value=2),
                                            starttime='90', identifier='\\delta',
                                            aspect_name=NormSystem.aspect_premise))

    # negative empowerment in premise ViVa aber nicht ViNo
    norm_system.add_norm(Norm(domain_aspect=NormAspectValues(
                             o_vals=(DimValue(name=o_scale[3], value=3),
                                     DimValue(name=o_scale[4], value=4)),
                             r_vals=(DimValue(name=r_scale[2], value=2),
                                     DimValue(name=r_scale[3], value=3)),
                             s_vals=(DimValue(name=s_scale[0], value=0),
                                     DimValue(name=s_scale[2], value=2)),
                             t_vals=(DimValue(name=t_scale[2], value=2),
                                     DimValue(name=t_scale[3], value=3)),
                             norm_type=NormSystem.type_prohib,
                             h_val=DimValue(name=h_scale[2], value=2),
                             starttime='90', identifier='\\eta',
                             aspect_name=NormSystem.aspect_domain),))
    norm_system.add_norm_premise_aspect(norm_identifier='\\eta',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[2], value=2)),
                                            s_vals=(DimValue(name=s_scale[2], value=2),
                                                    DimValue(name=s_scale[5], value=5)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_empow_pos,
                                            h_val=DimValue(name=h_scale[2], value=2),
                                            starttime='90', identifier='\\eta',
                                            aspect_name=NormSystem.aspect_premise))

    norm_system.add_norm_premise_aspect(norm_identifier='\\eta',
                                        premise_aspect=NormAspectValues(
                                            o_vals=(DimValue(name=o_scale[2], value=2),
                                                    DimValue(name=o_scale[5], value=5)),
                                            r_vals=(DimValue(name=r_scale[0], value=0),
                                                    DimValue(name=r_scale[2], value=2)),
                                            s_vals=(DimValue(name=s_scale[2], value=2),
                                                    DimValue(name=s_scale[3], value=3)),
                                            t_vals=(DimValue(name=t_scale[3], value=3),
                                                    DimValue(name=t_scale[3], value=3)),
                                            norm_type=NormSystem.type_empow_neg,
                                            h_val=DimValue(name=h_scale[2], value=2),
                                            starttime='90', identifier='\\eta',
                                            aspect_name=NormSystem.aspect_premise))

    saveid = 'subsumption'
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_premise)
    save_figures(norm_system=norm_system, saveid=saveid, aspect_name=NormSystem.aspect_domain)


if __name__ == "__main__":
    domain_scales = (o_scale, r_scale, s_scale,t_scale, h_scale)
    premise_scales = (o_scale, r_scale, s_scale,t_scale, h_scale)
    main_norm_system = NormSystem(domain_scales=domain_scales,premise_scales=premise_scales)
    subsumption(norm_system=main_norm_system)
    main_norm_system.reset()
    main_norm_system.change_color(NormSystem.type_obl, '#a0978d', NormSystem.aspect_domain)
    outside(norm_system=main_norm_system)
    print('Done with examples')
