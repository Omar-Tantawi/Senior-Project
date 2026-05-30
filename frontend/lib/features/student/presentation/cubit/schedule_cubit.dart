import 'package:first_try/features/student/data/mocks/student_mock_data.dart';
import 'package:first_try/features/student/data/repos/student_repo.dart';
import 'package:first_try/features/student/presentation/cubit/schedule_state.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ScheduleCubit extends Cubit<ScheduleState> {
  final StudentRepo repo;

  ScheduleCubit({required this.repo}) : super(ScheduleInitial());

  Future<void> load() async {
    emit(ScheduleLoading());
    try {
      final slots = await repo.getSchedule();
      emit(ScheduleLoaded(slots: slots, selectedDay: _todayKey()));
    } catch (_) {
      emit(ScheduleLoaded(
        slots: StudentMockData.schedule,
        selectedDay: _todayKey(),
      ));
    }
  }

  void selectDay(String day) {
    final s = state;
    if (s is ScheduleLoaded) {
      emit(s.copyWith(selectedDay: day));
    }
  }

  /// Returns the lowercase weekday name for today, snapping to the next
  /// school day (Sunday) if today is Fri or Sat — the school week is Sun–Thu.
  String _todayKey() {
    const keys = [
      'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    ];
    const schoolDays = {
      'sunday', 'monday', 'tuesday', 'wednesday', 'thursday',
    };
    final today = keys[(DateTime.now().weekday - 1).clamp(0, 6)];
    return schoolDays.contains(today) ? today : 'sunday';
  }
}
