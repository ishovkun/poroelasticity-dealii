#include <deal.II/base/parameter_handler.h>
#include <boost/algorithm/string.hpp>


namespace input_data {
  using namespace dealii;


  template<typename T>
  std::vector<T> parse_string_list(std::string list_string,
                                   std::string delimiter = ",")
    {
      std::vector<T> list;
      T item;
      if (list_string.size() == 0) return list;
      std::vector<std::string> strs;
      boost::split(strs, list_string, boost::is_any_of(delimiter));

      for (const auto &string_item : strs){
        std::stringstream convert(string_item);
        convert >> item;
        list.push_back(item);
      }
      return list;
    }


  template <int dim> class InputDataPoroel
  {
    // methods
  public:
    InputDataPoroel();
    void read_input_file(std::string input_file_name_);
  private:
    void declare_parameters();
    void assign_parameters();
    void compute_derived_parameters();
    void check_data();
    /* std::vector<int> parse_boundary_labels(std::string); */

    // variables
  private:
    ParameterHandler prm;
    std::string      input_file_name;

  public:
    // Equation data
    double perm, poro, visc, f_comp;
    double youngs_modulus, poisson_ratio, biot_coef;
    double bulk_density;
    double r_well, flow_rate;
    // Solver control
    double time_step, t_max;
    double fss_tol, pressure_tol;
    int max_fss_iterations, max_pressure_iterations;
    // In situ
    double p_init;
    std::vector<int>    stress_boundary_labels, displacement_boundary_labels;
    std::vector<int> stress_boundary_components, displacement_boundary_components;
    std::vector<double> stress_boundary_values, displacement_boundary_values;

    // Derived equation parameters
    double lame_constant, shear_modulus, bulk_modulus,
           grain_bulk_modulus, n_modulus, m_modulus;

  };  // End of class declaration


  template <int dim>
  InputDataPoroel<dim>::InputDataPoroel()
  {}  // EOM


  template <int dim>
    void InputDataPoroel<dim>::read_input_file(std::string input_file_name_)
  {
    input_file_name = input_file_name_;
    declare_parameters();
    prm.read_input(input_file_name);
    prm.print_parameters (std::cout, ParameterHandler::Text);
    assign_parameters();
    compute_derived_parameters();
    check_data();
  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::declare_parameters()
  {
    /* prm.enter_subsection("Mesh"); */
    /* prm.leave_subsection(); */

    { // properties section
      prm.enter_subsection("Properties");
      prm.declare_entry("Young modulus", "7e9", Patterns::Double(1));
      prm.declare_entry("Poisson ratio", "0.3", Patterns::Double(0, 0.5));
      prm.declare_entry("Biot coefficient", "0.9", Patterns::Double(0.1, 1));
      prm.declare_entry("Permeability", "1", Patterns::Double(1e-20, 1e5));
      prm.declare_entry("Porosity", "0.3", Patterns::Double(1e-5, 0.99999));
      prm.declare_entry("Viscosity", "1e-3", Patterns::Double(1e-6, 1));
      prm.declare_entry("Bulk density", "2700", Patterns::Double(5e2, 1e4));
      prm.declare_entry("Fluid compressibility", "45.8e-11", Patterns::Double(1e-16, 1e-2));
      prm.declare_entry("Well radius", "0.1", Patterns::Double(1e-2));
      prm.declare_entry("Flow rate", "1e-6", Patterns::Double());
      prm.leave_subsection();
    }
    { // In situ section - BC's & IC's
      prm.enter_subsection("In situ");
      prm.declare_entry("Initial pressure", "10e6", Patterns::Double(0));
      // Stress (neumann) boundaries
      prm.declare_entry("Stress boundary labels", "",
                        Patterns::List(Patterns::Integer()));
      prm.declare_entry("Stress boundary components", "",
                        Patterns::List(Patterns::Integer(0, 2)));
      prm.declare_entry("Stress boundary values", "",
                        Patterns::List(Patterns::Double()));
      // Displacement (dirichlet) boundaries
      prm.declare_entry("Displacement boundary labels", "0, 2, 3, 1",
                        Patterns::List(Patterns::Integer()));
      prm.declare_entry("Displacement boundary components", "1, 1, 0, 0",
                        Patterns::List(Patterns::Integer(0, 2)));
      prm.declare_entry("Displacement boundary values", "0, 0, 0, -0.1",
                      Patterns::List(Patterns::Double()));
      prm.leave_subsection();
    }
    { // solver section
      prm.enter_subsection("Solver");
      prm.declare_entry("Time step", "60", Patterns::Double(1e-8));
      prm.declare_entry("Time max", "60", Patterns::Double(1e-8));
      prm.declare_entry("Max FSS iterations", "50", Patterns::Integer(1, 1000));
      prm.declare_entry("Max pressure iterations", "50", Patterns::Integer(1, 1000));
      prm.declare_entry("FSS tolerance", "1e-8", Patterns::Double(1e-20, 1e-1));
      prm.declare_entry("Pressure tolerance", "1e-8", Patterns::Double(1e-20, 1e-1));
      /* prm.declare_entry("Displacement FE degree", "2", Patterns::Integer(1, 4)); */
      /* prm.declare_entry("Pressure FE degree", "2", Patterns::Integer(1, 4)); */
      prm.leave_subsection();
    }

  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::assign_parameters()
  {
    { // Properties section
      double mili_darcy = 9.869233e-16;
      prm.enter_subsection("Properties");
      youngs_modulus = prm.get_double("Young modulus");
      poisson_ratio = prm.get_double("Poisson ratio");
      biot_coef = prm.get_double("Biot coefficient");
      perm = prm.get_double("Permeability");
      perm *= mili_darcy;
      poro = prm.get_double("Porosity");
      visc = prm.get_double("Viscosity");
      bulk_density = prm.get_double("Bulk density");
      f_comp = prm.get_double("Fluid compressibility");
      r_well = prm.get_double("Well radius");
      flow_rate = prm.get_double("Flow rate");
      prm.leave_subsection();
    }
    { // In situ section
      prm.enter_subsection("In situ");
      p_init = prm.get_double("Initial pressure");
      // Stress (neumann) boundaries
      stress_boundary_labels =
        parse_string_list<int>(prm.get("Stress boundary labels"));
      stress_boundary_components =
        parse_string_list<int>(prm.get("Stress boundary components"));
      stress_boundary_values =
        parse_string_list<double>(prm.get("Stress boundary values"));
      // Displacement (dirichlet) boundaries
      displacement_boundary_labels =
        parse_string_list<int>(prm.get("Displacement boundary labels"));
      displacement_boundary_components =
        parse_string_list<int>(prm.get("Displacement boundary components"));
      displacement_boundary_values =
        parse_string_list<double>(prm.get("Displacement boundary values"));

      // debug with
      /* for (auto &item: stress_boundary_components) */
      /*   std::cout << item << std::endl; */
      prm.leave_subsection();
    }
    { // Solver section
      prm.enter_subsection("Solver");
      time_step = prm.get_double("Time step");
      t_max = prm.get_double("Time max");
      fss_tol = prm.get_double("FSS tolerance");
      pressure_tol = prm.get_double("Pressure tolerance");
      max_fss_iterations = prm.get_integer("Max FSS iterations");
      max_pressure_iterations = prm.get_integer("Max pressure iterations");
      prm.leave_subsection();
    }
  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::compute_derived_parameters()
  {
    double E = youngs_modulus, nu = poisson_ratio;
    lame_constant = E*nu/((1. + nu)*(1. - 2.*nu));
    shear_modulus = 0.5*E/(1 + nu);
    bulk_modulus = lame_constant + 2./3.*shear_modulus;
    grain_bulk_modulus = bulk_modulus/(1. - biot_coef);
    n_modulus = grain_bulk_modulus/(biot_coef - poro);
    m_modulus = (n_modulus/f_comp)/(n_modulus*poro + 1./f_comp);
  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::check_data()
  {
    double mili_darcy = 9.869233e-16;
    /* Assert(perm < 10e3 && perm > 0, ) */
    std::cout << "perm: " << perm/mili_darcy << " mD" << std::endl;
    /* std::cout << "Params: " << std::endl */
    /*           << "f_comp " << f_comp << std::endl */
    /*           << "perm " << perm << std::endl */
    /*           << "lambda " << lame_constant << std::endl */
    /*           << "G " << shear_modulus << std::endl */
    /*           << "K " << bulk_modulus << std::endl */
    /*           << "Ks " << grain_bulk_modulus << std::endl */
    /*           << "N " << n_modulus << std::endl */
    /*           << "M " << m_modulus << std::endl; */

  }  // EOM

}  // end of namespace
