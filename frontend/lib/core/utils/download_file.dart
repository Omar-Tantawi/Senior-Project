// Conditional import — picks the right implementation at compile time.
export 'download_file_io.dart'
    if (dart.library.html) 'download_file_web.dart';
