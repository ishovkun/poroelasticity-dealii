
namespace parse_command_line {
  using namespace std;

  string parse_command_line(int argc, char *const *argv) {
    string filename;
    if (argc < 2) {
      cout << "specify the file name" << endl;
      exit(1);
    }


    std::list<std::string> args;
    for (int i=1; i<argc; ++i)
      args.push_back(argv[i]);

    int arg_number = 1;
    while (args.size()){
      cout << args.front() << endl;
      if (arg_number == 1)
        filename = args.front();
      args.pop_front();
      arg_number++;
    } // EO while args

    return filename;
  }  // EOM

}  // end of namespace
