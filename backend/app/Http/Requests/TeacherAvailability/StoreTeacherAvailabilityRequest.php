<?php

namespace App\Http\Requests\TeacherAvailability;

use Illuminate\Foundation\Http\FormRequest;

class StoreTeacherAvailabilityRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'teacher_id'       => 'sometimes|exists:teachers,id',
            'dayofweek'        => 'required|in:Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday',
            'start_time'       => 'required|date_format:H:i',
            'end_time'         => 'required|date_format:H:i|after:start_time',
            'availabilitytype' => 'required|in:available,unavailable,preferred',
        ];
    }
}
