#include <deal.II/base/exceptions.h>
#include <vector>

namespace indexing {

  template <int dim> class TensorIndexer {
  public:
    TensorIndexer();
    int entryIndex(int tensor_index);
    std::vector<int>
      entryIndex(std::vector<int> tensor_indexes);
    int tensorIndex(int component);
  private:
    std::vector<int> tensor_to_entry_index_map;
  };

  // ----------------------- IMPLEMENTATION --------------------------------
  template <int dim> TensorIndexer<dim>::TensorIndexer()
    {
      switch (dim) {
      case 1:
        tensor_to_entry_index_map = {0};
        break;
      case 2:
        tensor_to_entry_index_map = {0, 1, 1, 2};
        break;
      case 3:
        tensor_to_entry_index_map = {0, 1, 2,
                                     1, 3, 4,
                                     2, 4, 5};
        break;
      default:
        Assert(false, dealii::ExcNotImplemented());
      }
    }

  template <int dim>
    int TensorIndexer<dim>::entryIndex(int tensor_index)
    {
      return tensor_to_entry_index_map[tensor_index];
    }

  template <int dim>
    std::vector<int> TensorIndexer<dim>::
    entryIndex(std::vector<int> tensor_indexes)
    {
      int n_comp = tensor_indexes.size();
      std::vector<int> entries(n_comp);
      for (int c=0; c<n_comp; ++c)
        entries[c] = tensor_to_entry_index_map[tensor_indexes[c]];
      return entries;
    }

} // end of namespace
